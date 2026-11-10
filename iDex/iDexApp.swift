//
//  iDexApp.swift
//  iDex
//
//  H.A.
//

import UIKit

@main
final class iDexApp: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        NotificationCenter.default.addObserver(self, selector: #selector(connectRestoredScene(_:)), name: UIScene.willConnectNotification, object: nil)
        return true
    }

    func application(_ application: UIApplication, configurationForConnecting session: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        let configuration = UISceneConfiguration(name: nil, sessionRole: session.role)
        configuration.delegateClass = DexSceneDelegate.self
        return configuration
    }

    // Existing installations can restore a session created by the previous
    // SwiftUI App lifecycle, bypassing configurationForConnecting entirely.
    @objc private func connectRestoredScene(_ notification: Notification) {
        guard let scene = notification.object as? UIWindowScene,
              scene.session.role == .windowApplication,
              !(scene.delegate is DexSceneDelegate) else { return }
        let delegate = DexSceneDelegate()
        scene.delegate = delegate
        delegate.connect(scene)
    }
}
