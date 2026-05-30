# Microgate VPN Configuration Guide
**Version:** 1.2 | **Classification:** Internal | **Department:** Engineering

## 1. Overview
All remote employees must connect via the corporate WireGuard VPN before accessing internal resources. The VPN gateway is hosted at vpn.microgate.it (IP: 93.184.216.34).

## 2. Client Setup
Download WireGuard from https://www.wireguard.com/install/. Import the configuration file provided by IT via encrypted email.

## 3. Network Topology
- Production subnet: 10.0.1.0/24
- Development subnet: 10.0.2.0/24
- Database cluster: 10.0.1.50 - 10.0.1.55
- ChromaDB instance: 10.0.1.60

## 4. Troubleshooting
If connection fails:
1. Verify your public key is registered with the VPN gateway
2. Check that UDP port 51820 is not blocked by your firewall
3. Contact helpdesk@microgate.it with your WireGuard client logs

## 5. Security Notes
- VPN sessions auto-expire after 12 hours of inactivity
- Split tunneling is disabled; all traffic routes through the VPN
- Access logs are retained for 90 days for audit purposes
