// Novel Formatter macOS Keychain adapter for OpenAI's official SIWC local SDK.
// The AES-256-GCM key is generated inside Keychain and never appears on argv/stdout.
import Foundation
import Security
import CryptoKit

let service = "org.novelformatter.siwc.encryption.v1"
let account = "local-credential-key"

func error(_ message: String) -> Never {
    fputs("Keychain unavailable: \(message)\n", stderr)
    exit(2)
}

func obtainKey() -> SymmetricKey {
    let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                kSecAttrService as String: service,
                                kSecAttrAccount as String: account,
                                kSecReturnData as String: true,
                                kSecMatchLimit as String: kSecMatchLimitOne]
    var result: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &result)
    if status == errSecSuccess {
        guard let data = result as? Data, data.count == 32 else { error("stored key invalid") }
        return SymmetricKey(data: data)
    }
    if status != errSecItemNotFound { error("read status \(status)") }
    var bytes = [UInt8](repeating: 0, count: 32)
    guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else { error("secure random failed") }
    let data = Data(bytes)
    var attrs = query
    attrs.removeValue(forKey: kSecReturnData as String)
    attrs.removeValue(forKey: kSecMatchLimit as String)
    attrs[kSecValueData as String] = data
    attrs[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
    let addStatus = SecItemAdd(attrs as CFDictionary, nil)
    if addStatus == errSecDuplicateItem { return obtainKey() }
    if addStatus != errSecSuccess { error("write status \(addStatus)") }
    return SymmetricKey(data: data)
}

let action = CommandLine.arguments.dropFirst().first ?? ""
guard ["seal", "open", "check"].contains(action) else { error("unknown operation") }
let key = obtainKey()
if action == "check" { print("ready"); exit(0) }
let input = FileHandle.standardInput.readDataToEndOfFile()
do {
    let output: Data
    if action == "seal" {
        guard let combined = try AES.GCM.seal(input, using: key).combined else { error("cannot encode sealed data") }
        output = combined
    } else {
        output = try AES.GCM.open(AES.GCM.SealedBox(combined: input), using: key)
    }
    FileHandle.standardOutput.write(output)
} catch { error("authentication/decryption failed") }
