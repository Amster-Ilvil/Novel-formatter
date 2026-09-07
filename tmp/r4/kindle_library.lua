local BookList = require("ui/widget/booklist")
local ButtonDialog = require("ui/widget/buttondialog")
local InfoMessage = require("ui/widget/infomessage")
local InputDialog = require("ui/widget/inputdialog")
local UIManager = require("ui/uimanager")
local filemanagerutil = require("apps/filemanager/filemanagerutil")
local logger = require("logger")
local _ = require("gettext")
local CoverBrowserExt = require("lua/coverbrowser_ext")
local T = require("ffi/util").template

local KindleLibrary = {}
KindleLibrary.__index = KindleLibrary

function KindleLibrary:new(virtual_library, cache_manager)
    return setmetatable({
        virtual_library = virtual_library,
        cache_manager = cache_manager,
        ui = nil,
        booklist_menu = nil,
        group_dialog = nil,
        return_to_library_request = nil,
        active_group = nil,
    }, self)
end

function KindleLibrary:setUI(ui) self.ui = ui end

function KindleLibrary:requestReturnToLibrary(origin_path)
    self.return_to_library_request = { origin_path = origin_path }
end

function KindleLibrary:takeReturnToLibraryRequest()
    local request = self.return_to_library_request
    self.return_to_library_request = nil
    return request
end

local function showInfo(text, timeout)
    UIManager:show(InfoMessage:new({ text = text, timeout = timeout or 4 }))
end

function KindleLibrary:close()
    if self.booklist_menu then
        UIManager:close(self.booklist_menu)
        self.booklist_menu = nil
    end
    if self.group_dialog then
        UIManager:close(self.group_dialog)
        self.group_dialog = nil
    end
end

function KindleLibrary:buildEntries(force, group)
    local entries, err = self.virtual_library:getBookEntries(force, group)
    if not entries then return nil, err end
    for _, item in ipairs(entries) do
        local book = self.virtual_library:getBook(item.kindle_book_id)
        if book and book.open_mode == "blocked" then
            item.text = item.text .. " [blocked]"
            item.mandatory = self.virtual_library:getBlockedReasonText(book)
        elseif book and book.open_mode == "convert" then
            local cache_state = self.virtual_library:isBookPrepared(book) and " · cached" or " · prepare on open"
            item.mandatory = item.mandatory .. cache_state
        end
    end
    return entries
end

local function groupTitle(group)
    if group == "amazon_jp" then return "日亚书库" end
    if group == "physical_downloads" then return "实际下载（新下载优先）" end
    if group == "physical_kfx" then return "实际 KFX（新下载优先）" end
    if group == "amazon_other" then return "Amazon 其它" end
    if group == "local" then return "本机其它书籍" end
    return "全部 Kindle 书籍"
end

function KindleLibrary:showGroupChooser(ui, force)
    self.ui = ui or self.ui
    if not self.ui then return false end
    self:close()
    self.active_group = nil

    local counts, err = self.virtual_library:getGroupCounts(force ~= false)
    if not counts then
        showInfo(_("Failed to build Kindle library:\n") .. (err or _("unknown error")))
        return false
    end
    if counts.all == 0 then
        showInfo(_("No Kindle books were found in the Kindle content catalog."))
        return false
    end

    local manager = self
    self.group_dialog = ButtonDialog:new({
        title = T("Kindle Library (%1)", counts.all),
        buttons = {
            {
                {
                    text = T("日亚书库 (%1)", counts.amazon_jp),
                    enabled = counts.amazon_jp > 0,
                    callback = function()
                        UIManager:close(manager.group_dialog)
                        manager.group_dialog = nil
                        manager:showBookGroup(manager.ui, false, "amazon_jp")
                    end,
                },
                {
                    text = T("实际下载 (%1)", counts.physical_downloads),
                    enabled = counts.physical_downloads > 0,
                    callback = function()
                        UIManager:close(manager.group_dialog)
                        manager.group_dialog = nil
                        manager:showBookGroup(manager.ui, false, "physical_downloads")
                    end,
                },
            },
            {
                {
                    text = T("实际 KFX (%1)", counts.physical_kfx),
                    enabled = counts.physical_kfx > 0,
                    callback = function()
                        UIManager:close(manager.group_dialog)
                        manager.group_dialog = nil
                        manager:showBookGroup(manager.ui, false, "physical_kfx")
                    end,
                },
                {
                    text = T("Amazon 其它 (%1)", counts.amazon_other),
                    enabled = counts.amazon_other > 0,
                    callback = function()
                        UIManager:close(manager.group_dialog)
                        manager.group_dialog = nil
                        manager:showBookGroup(manager.ui, false, "amazon_other")
                    end,
                },
            },
            {
                {
                    text = T("本机其它 (%1)", counts.local_books),
                    enabled = counts.local_books > 0,
                    callback = function()
                        UIManager:close(manager.group_dialog)
                        manager.group_dialog = nil
                        manager:showBookGroup(manager.ui, false, "local")
                    end,
                },
                {
                    text = T("全部 (%1)", counts.all),
                    callback = function()
                        UIManager:close(manager.group_dialog)
                        manager.group_dialog = nil
                        manager:showBookGroup(manager.ui, false, "all")
                    end,
                },
            },
        },
    })
    UIManager:show(self.group_dialog)
    return true
end

function KindleLibrary:show(ui, force)
    return self:showGroupChooser(ui, force)
end

function KindleLibrary:showBookGroup(ui, force, group)
    self.ui = ui or self.ui
    if not self.ui then return false end
    if self.booklist_menu then
        UIManager:close(self.booklist_menu)
        self.booklist_menu = nil
    end
    self.active_group = group or "all"

    local entries, err = self:buildEntries(force ~= false, self.active_group)
    if not entries then
        showInfo(_("Failed to build Kindle library:\n") .. (err or _("unknown error")))
        return false
    end
    if #entries == 0 then
        showInfo("这个分组里没有书。")
        return self:showGroupChooser(self.ui, false)
    end

    local manager = self
    self.booklist_menu = BookList:new({
        name = "kindle_library",
        title = groupTitle(self.active_group),
        title_bar_left_icon = "appbar.menu",
        onLeftButtonTap = function()
            if manager.booklist_menu then
                UIManager:close(manager.booklist_menu)
                manager.booklist_menu = nil
            end
            manager:showGroupChooser(manager.ui, false)
        end,
        onMenuSelect = function(_, item) return manager:openItem(item) end,
        onMenuHold = function(_, item) return manager:showBookDialog(item) end,
        ui = self.ui,
        _manager = self,
        _recreate_func = function()
            manager:showBookGroup(manager.ui, true, manager.active_group)
        end,
    })
    if CoverBrowserExt.apply(self.booklist_menu) then
        logger.info("KindlePlugin: Kindle Library uses a CoverBrowser display mode")
    end
    self.booklist_menu.close_callback = function()
        if manager.booklist_menu then
            UIManager:close(manager.booklist_menu)
            manager.booklist_menu = nil
        end
        manager:showGroupChooser(manager.ui, false)
    end
    self.booklist_menu:switchItemTable(T("%1 (%2)", groupTitle(self.active_group), #entries), entries, -1)
    UIManager:show(self.booklist_menu)
    return true
end

function KindleLibrary:openItem(item)
    local book = item and self.virtual_library:getBook(item.kindle_book_id)
    if not book then
        showInfo(_("Book entry is no longer available."))
        return true
    end
    if book.open_mode == "blocked" then
        showInfo(self.virtual_library:getBlockedReasonText(book))
        return true
    end
    if not book.source_path then
        showInfo(self.virtual_library:getBlockedReasonText({ block_reason = "missing_source" }))
        return true
    end

    logger.info("KindlePlugin: requesting native open for:", book.source_path)
    local close_callback = self.booklist_menu and self.booklist_menu.close_callback or nil
    filemanagerutil.openFile(self.ui, book.source_path, function()
        local file_chooser = self.ui and self.ui.file_chooser
        self:requestReturnToLibrary(file_chooser and file_chooser.path or nil)
        if close_callback then close_callback() end
    end)
    return true
end

function KindleLibrary:editChineseTitle(book)
    local dialog
    dialog = InputDialog:new({
        title = "设置中文书名（留空恢复自动名称）",
        input = self.virtual_library:getTitleAlias(book) or self.virtual_library:getDisplayTitle(book) or "",
        buttons = {
            {
                {
                    text = _("Cancel"),
                    id = "close",
                    callback = function() UIManager:close(dialog) end,
                },
                {
                    text = _("Save"),
                    is_enter_default = true,
                    callback = function()
                        local value = dialog:getInputText() or ""
                        self.virtual_library:setTitleAlias(book, value)
                        UIManager:close(dialog)
                        self:showBookGroup(self.ui, true, self.active_group or "all")
                    end,
                },
            },
        },
    })
    UIManager:show(dialog)
    dialog:onShowKeyboard()
end

function KindleLibrary:showBookDialog(item)
    local book = item and self.virtual_library:getBook(item.kindle_book_id)
    if not book then return true end

    local display = self.virtual_library:getDisplayTitle(book)
    local details = display
        .. "\n原始书名: " .. tostring(book.original_title or book.title or "")
        .. "\n格式: " .. tostring(book.format_label or "?")
        .. "\n语言: " .. tostring(book.language or "")
        .. "\nASIN/CDE: " .. tostring(book.cde_key or "")
        .. "\n实际下载: " .. (book.filesystem_download and "是" or "否")
        .. "\n实际KFX: " .. (book.filesystem_kfx and "是" or "否")
        .. "\n\n" .. tostring(book.source_path or _("Cloud-only Kindle entry"))
    if book.open_mode == "blocked" then
        details = details .. "\n\n" .. self.virtual_library:getBlockedReasonText(book)
    end

    local dialog
    dialog = ButtonDialog:new({
        title = details,
        buttons = {
            {
                {
                    text = _("Open"),
                    callback = function()
                        UIManager:close(dialog)
                        self:openItem(item)
                    end,
                    enabled = book.open_mode ~= "blocked",
                },
                {
                    text = "中文书名",
                    callback = function()
                        UIManager:close(dialog)
                        self:editChineseTitle(book)
                    end,
                },
            },
            {
                {
                    text = _("Refresh"),
                    callback = function()
                        UIManager:close(dialog)
                        self.virtual_library:refresh(true)
                        self:showBookGroup(self.ui, false, self.active_group or "all")
                    end,
                },
                {
                    text = _("Clear Cache"),
                    callback = function()
                        UIManager:close(dialog)
                        if self.cache_manager then
                            local ok, err = self.cache_manager:clearBookCache(book)
                            if not ok then
                                showInfo(_("Failed to clear cache:\n") .. (err or _("unknown error")))
                                return
                            end
                        end
                        self:showBookGroup(self.ui, false, self.active_group or "all")
                    end,
                    enabled = book.open_mode ~= "direct",
                },
            },
            {
                {
                    text = _("Show Info"),
                    callback = function()
                        UIManager:close(dialog)
                        showInfo(details, 8)
                    end,
                },
            },
        },
    })
    UIManager:show(dialog)
    return true
end

return KindleLibrary
