local BD = require("ui/bidi")
local Device = require("device")
local logger = require("logger")
local util = require("util")
local _ = require("gettext")

local VirtualLibrary = {}
VirtualLibrary.__index = VirtualLibrary

VirtualLibrary.VIRTUAL_LIBRARY_NAME = "Kindle Library"
VirtualLibrary.KINDLE_DOCUMENTS_ROOT = "/mnt/us/documents"

function VirtualLibrary:new(library_index)
    local instance = {
        library_index = library_index,
        settings = {},
        cache_manager = nil,
        books_by_id = {},
        books_by_real_path = {},
        mapping_attempted = false,
    }
    return setmetatable(instance, self)
end

function VirtualLibrary:setSettings(settings)
    self.settings = settings or {}
    self.settings.title_aliases = self.settings.title_aliases or {}
end

function VirtualLibrary:setCacheManager(cache_manager)
    self.cache_manager = cache_manager
end

local function isPathWithin(path, root)
    if type(path) ~= "string" or type(root) ~= "string" or root == "" then
        return false
    end
    return path == root or path:sub(1, #root + 1) == root .. "/"
end

local function isKindleSourcePath(path)
    if type(path) ~= "string" then
        return false
    end
    local extension = path:lower():match("%.([%w]+)$")
    return extension == "kfx" or extension == "azw" or extension == "azw3" or extension == "mobi"
end

local function sanitizeDisplayName(name)
    local cleaned = tostring(name or "Untitled"):gsub("[/\\]+", " "):gsub("%s+", " ")
    cleaned = cleaned:gsub("^%s+", ""):gsub("%s+$", "")
    return cleaned ~= "" and cleaned or "Untitled"
end

function VirtualLibrary:getDisplayTitle(book)
    if not book then
        return "Untitled"
    end
    local aliases = self.settings.title_aliases or {}
    local alias = aliases[book.id] or (book.cde_key ~= "" and aliases[book.cde_key] or nil)
    if type(alias) == "string" and alias:match("%S") then
        return sanitizeDisplayName(alias)
    end
    return sanitizeDisplayName(book.auto_display_name or book.display_name or book.title or book.id)
end

function VirtualLibrary:getTitleAlias(book)
    if not book then
        return nil
    end
    local aliases = self.settings.title_aliases or {}
    return aliases[book.id] or (book.cde_key ~= "" and aliases[book.cde_key] or nil)
end

function VirtualLibrary:setTitleAlias(book, value)
    if not book then
        return false
    end
    self.settings.title_aliases = self.settings.title_aliases or {}
    local cleaned = sanitizeDisplayName(value or "")
    if value == nil or tostring(value):match("^%s*$") then
        self.settings.title_aliases[book.id] = nil
        if book.cde_key and book.cde_key ~= "" then
            self.settings.title_aliases[book.cde_key] = nil
        end
    else
        self.settings.title_aliases[book.id] = cleaned
        if book.cde_key and book.cde_key ~= "" then
            self.settings.title_aliases[book.cde_key] = cleaned
        end
    end
    G_reader_settings:saveSetting("kindle_plugin", self.settings)
    return true
end

function VirtualLibrary:buildMappings(force)
    self.mapping_attempted = true
    local books, err = self.library_index:getBooks(force)
    if not books then
        return nil, err
    end

    self.books_by_id = {}
    self.books_by_real_path = {}
    for _, book in ipairs(books) do
        self.books_by_id[book.id] = book
        if book.source_path then
            self.books_by_real_path[book.source_path] = book
        end
        if self.cache_manager then
            local cached_path = self.cache_manager:getCachePaths(book)
            if cached_path then
                self.books_by_real_path[cached_path] = book
            end
        end
    end
    logger.info("KindlePlugin: built real-path mappings for", #books, "books")
    return books
end

function VirtualLibrary:refresh(force)
    return self:buildMappings(force)
end

function VirtualLibrary:isActive()
    return self.settings.enable_virtual_library ~= false
end

function VirtualLibrary:getBook(path_or_id)
    if not path_or_id then
        return nil
    end
    local book = self.books_by_id[path_or_id] or self.books_by_real_path[path_or_id]
    if book then
        return book
    end

    local cache_dir = self.settings.cache_dir
    if not cache_dir and self.cache_manager and self.cache_manager.getCacheDir then
        cache_dir = self.cache_manager:getCacheDir()
    end
    local should_build = type(path_or_id) == "string"
        and (
            path_or_id:match("^cc:")
            or path_or_id:match("^sha1:")
            or path_or_id:match("^fs:")
            or isPathWithin(path_or_id, cache_dir)
            or isPathWithin(path_or_id, self.KINDLE_DOCUMENTS_ROOT)
            or isKindleSourcePath(path_or_id)
        )
    if not self.mapping_attempted and should_build then
        local books = self:buildMappings(false)
        if books then
            return self.books_by_id[path_or_id] or self.books_by_real_path[path_or_id]
        end
    end
    return nil
end

function VirtualLibrary:getBlockedReasonText(book)
    local reason = book and book.block_reason or "conversion_failed"
    local text = {
        drm = _("This DRM-protected Kindle format is not supported."),
        missing_source = _("The source file is missing."),
        unsupported_format = _("This Kindle file format is not supported."),
        conversion_failed = _("Failed to prepare this book for reading."),
        drm_extractor_unavailable = _(
            "This Kindle firmware cannot extract this book's access key by itself. "
                .. "Install a compatible kfxdedrm native extractor, then reopen the book."
        ),
        drm_key_extraction_failed = _("Could not extract this book's access key. Check the KOReader debug log for details."),
        native_non_kfx = _("This is a physical Kindle download, but it is not KFX. The KFX-to-EPUB backend does not convert this format."),
        drm_after_key_extraction = _(
            "A book access key was extracted, but the book still could not be decrypted. "
                .. "Try re-downloading the book in the Kindle reader and opening it again."
        ),
    }
    return text[reason] or _("This book cannot be opened yet.")
end

function VirtualLibrary:isBookPrepared(book)
    if not book then return false end
    if book.open_mode == "direct" then return book.source_path ~= nil end
    if book.open_mode == "blocked" or not self.cache_manager then return false end
    return self.cache_manager:isFresh(book) == true
end

function VirtualLibrary:resolveBookPath(book)
    if not book then return nil, "missing book" end
    if book.open_mode == "blocked" then
        return nil, book.block_reason or "conversion_failed"
    end
    if book.open_mode == "direct" then return book.source_path end
    if not self.cache_manager then return nil, "conversion_failed" end
    local cached_path, err = self.cache_manager:ensureCachedEpub(book)
    if cached_path then
        self.books_by_real_path[cached_path] = book
        return cached_path
    end
    return nil, err or "conversion_failed"
end

function VirtualLibrary:createVirtualFolderEntry(parent_path)
    local entry = {
        text = self.VIRTUAL_LIBRARY_NAME .. "/",
        path = parent_path or Device.home_dir or "/",
        attr = { mode = "directory" },
        is_kindle_library_folder = true,
        bidi_wrap_func = BD.directory,
    }
    if self.settings.virtual_library_cover_path and self.settings.virtual_library_cover_path ~= "" then
        entry.pt_cover_path = self.settings.virtual_library_cover_path
    end
    return entry
end

function VirtualLibrary:getGroupCounts(force)
    local books, err = self:buildMappings(force)
    if not books then return nil, err end
    local counts = { amazon_jp = 0, amazon_other = 0, local_books = 0, physical_downloads = 0, physical_kfx = 0, all = #books }
    for _, book in ipairs(books) do
        if book.filesystem_download then counts.physical_downloads = counts.physical_downloads + 1 end
        if book.filesystem_kfx then counts.physical_kfx = counts.physical_kfx + 1 end
        if book.library_group == "amazon_jp" then
            counts.amazon_jp = counts.amazon_jp + 1
        elseif book.library_group == "amazon_other" then
            counts.amazon_other = counts.amazon_other + 1
        else
            counts.local_books = counts.local_books + 1
        end
    end
    return counts
end

local function groupMatches(book, group)
    if not group or group == "all" then return true end
    if group == "local" then return book.library_group == "local" end
    if group == "physical_downloads" then return book.filesystem_download == true end
    if group == "physical_kfx" then return book.filesystem_kfx == true end
    return book.library_group == group
end

function VirtualLibrary:getBookEntries(force, group)
    local books, err = self:buildMappings(force)
    if not books then return nil, err end

    local entries = {}
    for _, book in ipairs(books) do
        if groupMatches(book, group) then
            local title = self:getDisplayTitle(book)
            local original = sanitizeDisplayName(book.original_title or book.title)
            local format_tag = book.format_label and ("[" .. book.format_label .. "] ") or ""
            local authors = book.authors and table.concat(book.authors, ", ") or ""
            local mandatory = authors ~= "" and authors or util.getFriendlySize(book.source_size or 0)
            if title ~= original and original ~= "Untitled" then
                mandatory = original .. (mandatory ~= "" and (" · " .. mandatory) or "")
            end
            local entry_file = book.source_path or ""
            if book.open_mode ~= "blocked" and book.open_mode ~= "direct" and self.cache_manager and self:isBookPrepared(book) then
                entry_file = self.cache_manager:getCachePaths(book) or entry_file
            end
            table.insert(entries, {
                text = format_tag .. title,
                file = entry_file,
                path = book.source_path or "",
                attr = { mode = "file", size = book.source_size or 0 },
                mandatory = mandatory,
                kindle_book_id = book.id,
            })
        end
    end
    table.sort(entries, function(a, b)
        if group == "physical_downloads" or group == "physical_kfx" then
            local left = self:getBook(a.kindle_book_id)
            local right = self:getBook(b.kindle_book_id)
            local lm = tonumber(left and left.source_mtime) or 0
            local rm = tonumber(right and right.source_mtime) or 0
            if lm ~= rm then return lm > rm end
        end
        return tostring(a.text):lower() < tostring(b.text):lower()
    end)
    return entries
end

return VirtualLibrary
