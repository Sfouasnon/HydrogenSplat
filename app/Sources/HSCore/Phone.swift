import Foundation

/// `hs phone` output, decoded.
public struct PhoneDevice: Identifiable, Hashable, Sendable {
    public var id: String { serial }
    public let serial: String
    public let state: String
    public let model: String?
    public let product: String?
    public var clips: [PhoneClip]

    public var ready: Bool { state == "device" }
}

public struct PhoneClip: Identifiable, Hashable, Sendable {
    public var id: String { serial + ":" + path }
    public let serial: String
    public let name: String
    public let path: String
    public let bytes: Int
    public let mtime: String
}

public enum PhoneListing {
    public static func devices(from events: [HSEvent]) -> [PhoneDevice] {
        var devs: [PhoneDevice] = []
        for e in events where e.kind == "metric" && e.stage == "phone" {
            if e.name == "devices", let arr = e.value?.array {
                devs = arr.compactMap { d in
                    guard let s = d["serial"]?.string else { return nil }
                    return PhoneDevice(serial: s, state: d["state"]?.string ?? "?",
                                       model: d["model"]?.string, product: d["product"]?.string, clips: [])
                }
            } else if e.name == "clips", let serial = e["serial"]?.string, let arr = e.value?.array,
                      let i = devs.firstIndex(where: { $0.serial == serial }) {
                devs[i].clips = arr.compactMap { c in
                    guard let n = c["name"]?.string, let p = c["path"]?.string else { return nil }
                    return PhoneClip(serial: serial, name: n, path: p, bytes: c["bytes"]?.int ?? 0,
                                     mtime: c["mtime"]?.string ?? "")
                }
            }
        }
        return devs
    }
}
