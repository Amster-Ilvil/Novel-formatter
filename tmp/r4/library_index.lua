local lfs = require("libs/libkoreader-lfs")
local logger = require("logger")

local LibraryIndex = {}
LibraryIndex.__index = LibraryIndex

function LibraryIndex:new(ccdb_scanner)
    local instance = {
        ccdb_scanner = ccdb_scanner,
        settings = {},
        books = {},
        loaded_at = 0,
    }
    setmetatable(instance, self)
    return instance
end

function LibraryIndex:setSettings(settings)
    self.settings = settings or {}
end

local function sortBooks(books)
    table.sort(books, function(left, right)
        local left_name = (left.display_name or left.title or left.source_path or ""):lower()
        local right_name = (right.display_name or right.title or right.source_path or ""):lower()
        if left_name == right_name then
            return (left.source_path or "") < (right.source_path or "")
        end
        return left_name < right_name
    end)
end

local function looksJapanese(text)
    text = tostring(text or "")
    for i = 1, #text - 2 do
        local a, b, c = text:byte(i, i + 2)
        if a == 0xE3 then
            if b == 0x81 and c and c >= 0x80 then
                return true
            end
            if b == 0x82 or b == 0x83 then
                return true
            end
        end
    end
    return false
end

local function basename(path)
    return tostring(path or ""):match("([^/]+)$") or ""
end

local function stripExtension(name)
    return tostring(name or ""):gsub("%.[^%.]+$", "")
end

local function parseAsin(path)
    local stem = stripExtension(basename(path))
    local token = stem:match("_([A-Z0-9]+)$") or stem:match("%-([A-Z0-9]+)$")
    if token and #token == 10 then
        return token
    end
    local last
    for candidate in stem:gmatch("([A-Z0-9]+)") do
        if #candidate == 10 then
            last = candidate
        end
    end
    return last
end

local function titleFromFilename(path)
    local stem = stripExtension(basename(path))
    local asin = parseAsin(path)
    if asin then
        stem = stem:gsub("_" .. asin .. "$", "")
        stem = stem:gsub("%-" .. asin .. "$", "")
        stem = stem:gsub(asin .. "$", "")
    end
    stem = stem:gsub("_+", " "):gsub("%s+", " ")
    stem = stem:gsub("^%s+", ""):gsub("%s+$", "")
    return stem ~= "" and stem or (asin or "Kindle book")
end

local function stableFsId(path)
    local hash = 5381
    path = tostring(path or "")
    for i = 1, #path do
        hash = (hash * 33 + path:byte(i)) % 4294967296
    end
    return string.format("fs:%08x", hash)
end

local function fileHasKfxSignature(path)
    local handle = io.open(path, "rb")
    if not handle then
        return false
    end
    local header = handle:read(8) or ""
    handle:close()
    return header:sub(1, 4) == "CONT" or header == "\234DRMION\238"
end

local function physicalFormat(path)
    local ext = tostring(path or ""):lower():match("%.([%w]+)$") or ""
    if ext == "kfx" then
        return "KFX", true
    end
    if ext == "azw8" then
        return "KFX", true
    end
    if ext == "azw" and fileHasKfxSignature(path) then
        return "KFX", true
    end
    if ext == "azw3" then
        return "AZW3", false
    end
    if ext == "azw" then
        return "AZW", false
    end
    if ext == "mobi" then
        return "MOBI", false
    end
    return nil, false
end

local function skipDirectory(name)
    local lower = tostring(name or ""):lower()
    return lower == "." or lower == ".." or lower:match("%.sdr$") ~= nil
        or lower == "kindle-epub" or lower == ".koreader"
end

local function discoverPhysicalBooks()
    local root = "/mnt/us/documents"
    if lfs.attributes(root, "mode") ~= "directory" then
        return {}
    end

    local found, seen = {}, {}
    local function walk(dir, depth)
        local ok, iter, state, var = pcall(lfs.dir, dir)
        if not ok or not iter then
            return
        end
        for name in iter, state, var do
            if name ~= "." and name ~= ".." then
                local path = dir .. "/" .. name
                local mode = lfs.attributes(path, "mode")
                if mode == "directory" then
                    if depth < 2 and not skipDirectory(name) then
                        walk(path, depth + 1)
                    end
                elseif mode == "file" and not seen[path] then
                    local format_label, is_kfx = physicalFormat(path)
                    if format_label then
                        seen[path] = true
                        local attr = lfs.attributes(path) or {}
                        local lower_path = path:lower()
                        table.insert(found, {
                            path = path,
                            size = tonumber(attr.size) or 0,
                            mtime = tonumber(attr.modification) or 0,
                            format_label = format_label,
                            is_kfx = is_kfx,
                            is_amazon_download = lower_path:find("/downloads/items01/", 1, true) ~= nil,
                        })
                    end
                end
            end
        end
    end

    walk(root, 0)
    table.sort(found, function(a, b)
        if a.mtime == b.mtime then
            return a.path < b.path
        end
        return a.mtime > b.mtime
    end)
    logger.info("KindlePlugin: AmazonJP R4 filesystem scan found", #found, "physical Kindle book files")
    return found
end

local function isAmazonCatalogBook(book)
    local cde_type = tostring(book.cde_type or ""):upper()
    local cde_key = tostring(book.cde_key or "")
    return cde_type == "EBOK" and #cde_key == 10 and cde_key:match("^[A-Z0-9]+$") ~= nil
end

local function mergePhysicalBooks(books)
    local by_path, by_asin = {}, {}
    for _, book in ipairs(books) do
        if book.source_path and book.source_path ~= "" then
            by_path[book.source_path] = book
        end
        local key = tostring(book.cde_key or "")
        if #key == 10 and key:match("^[A-Z0-9]+$") then
            by_asin[key] = by_asin[key] or book
        end
        if book.library_group == "amazon_other"
            and isAmazonCatalogBook(book)
            and looksJapanese(book.original_title or book.title or book.display_name)
        then
            book.library_group = "amazon_jp"
        end
    end

    local discovered = discoverPhysicalBooks()
    for _, item in ipairs(discovered) do
        local path = item.path
        local asin = parseAsin(path)
        local file_title = titleFromFilename(path)
        local book = by_path[path] or (asin and by_asin[asin]) or nil

        if book then
            if book.source_path ~= path then
                if book.source_path and book.source_path ~= "" then
                    by_path[book.source_path] = nil
                end
                book.source_path = path
                by_path[path] = book
            end
            book.filesystem_download = item.is_amazon_download
            book.filesystem_kfx = item.is_kfx
            book.physical_format_label = item.format_label
            book.source_mtime = item.mtime
            book.source_size = item.size > 0 and item.size or book.source_size
            if item.is_kfx then
                book.open_mode = "convert"
                book.block_reason = nil
                book.format_label = "KFX"
            elseif not book.format_label or book.format_label == "?" then
                book.format_label = item.format_label
            end
            if (not book.cde_key or book.cde_key == "") and asin then
                book.cde_key = asin
            end
            if (not book.original_title or book.original_title == "" or book.original_title == "Untitled") and file_title ~= "" then
                book.original_title = file_title
                book.title = file_title
            end
            if (not book.auto_display_name or book.auto_display_name == "" or book.auto_display_name == "Untitled") and file_title ~= "" then
                book.auto_display_name = file_title
                book.display_name = file_title
            end
            if book.library_group ~= "amazon_jp" and item.is_amazon_download and looksJapanese(file_title) then
                book.library_group = "amazon_jp"
            end
        else
            local group = (item.is_amazon_download and looksJapanese(file_title)) and "amazon_jp"
                or (item.is_amazon_download and "amazon_other" or "local")
            book = {
                id = stableFsId(path),
                source_path = path,
                title = file_title,
                original_title = file_title,
                auto_display_name = file_title,
                display_name = file_title,
                authors = {},
                language = looksJapanese(file_title) and "ja" or "",
                cde_key = asin or "",
                cde_type = asin and "EBOK" or "",
                mime_type = item.is_kfx and "application/x-kfx-ebook" or "",
                format_label = item.format_label,
                physical_format_label = item.format_label,
                library_group = group,
                open_mode = item.is_kfx and "convert" or "blocked",
                block_reason = item.is_kfx and nil or "native_non_kfx",
                source_size = item.size,
                source_mtime = item.mtime,
                filesystem_download = item.is_amazon_download,
                filesystem_kfx = item.is_kfx,
            }
            table.insert(books, book)
            by_path[path] = book
            if asin then
                by_asin[asin] = by_asin[asin] or book
            end
        end
    end

    return books
end

function LibraryIndex:scan()
    if not self.ccdb_scanner then
        local ok, CcDbScanner = pcall(require, "lua/ccdb_scanner")
        if ok then
            self.ccdb_scanner = CcDbScanner:new()
        end
    end

    local books = {}
    if self.ccdb_scanner and self.ccdb_scanner:isAvailable() then
        logger.info("KindlePlugin: scanning library via cc.db")
        local catalog_books, err = self.ccdb_scanner:scan()
        if catalog_books then
            books = catalog_books
        else
            logger.warn("KindlePlugin: cc.db scan failed; continuing with physical-file fallback:", err)
        end
    else
        logger.warn("KindlePlugin: Kindle content catalog unavailable; using physical-file fallback")
    end

    books = mergePhysicalBooks(books)
    sortBooks(books)
    return books
end

function LibraryIndex:refresh(force)
    local ttl = tonumber(self.settings.index_ttl_seconds) or 300
    if not force and (os.time() - self.loaded_at) < ttl and #self.books > 0 then
        return self.books
    end

    local books, err = self:scan()
    if not books then
        return nil, err
    end

    self.books = books
    self.loaded_at = os.time()
    return books
end

function LibraryIndex:getBooks(force)
    return self:refresh(force)
end

return LibraryIndex
