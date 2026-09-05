#ifndef MyAppVersion
  #define MyAppVersion "dev"
#endif
#ifndef SourceDir
  #error SourceDir must be supplied by the build script.
#endif
#ifndef OutputDir
  #error OutputDir must be supplied by the build script.
#endif

[Setup]
AppId={{8B7E42DA-48EE-4E97-9A6C-13A0F36C9654}
AppName=Novel Formatter Studio
AppVersion={#MyAppVersion}
AppVerName=Novel Formatter Studio {#MyAppVersion}
AppPublisher=Amster-Ilvil
AppPublisherURL=https://github.com/Amster-Ilvil/Novel-formatter
AppSupportURL=https://github.com/Amster-Ilvil/Novel-formatter/issues
DefaultDirName={localappdata}\Programs\Novel Formatter Studio
DefaultGroupName=Novel Formatter Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputDir}
OutputBaseFilename=NovelFormatter_{#MyAppVersion}_Windows_x64_Setup
SetupIconFile={#SourceDir}\icon.ico
UninstallDisplayIcon={app}\icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=no
RestartApplications=no
ChangesEnvironment=no
UsePreviousAppDir=yes
UsePreviousTasks=yes
Uninstallable=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional options:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Novel Formatter Studio"; Filename: "{app}\启动Windows.bat"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\Novel Formatter Studio"; Filename: "{app}\启动Windows.bat"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\启动Windows.bat"; Description: "Launch Novel Formatter Studio"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent shellexec
