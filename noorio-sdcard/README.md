# Noorio SD Card

My test device has been a 'Noorio 2K Indoor Security Camera' (Amazon [link](https://www.amazon.com/dp/B0C8J39JHH)) which at the time of writing was available for under $20 delivered.

After cracking it open and dumping the stock firmware, it was discovered that it runs `/mnt/sdcard/factorytest/factorytest.sh` on boot, using root permissions. This is super useful because it lets us change the root password, kill the stock firmware (LeCam), and install dropbear for an SSH shell.

## Use

```sh
python3 sdcard/make_card.py
```

It asks for four things:

| prompt | goes into |
|---|---|
| hostname | `/etc/hostname`, and the DHCP request, so your router shows a name |
| wifi SSID | `/etc/conf/wpa.conf`, as hex |
| wifi passphrase | the same file, as a PBKDF2-SHA1 pre-shared key |
| root password | `/etc/shadow`, as an MD5-crypt hash |

Then:

1. Format an SD card as FAT and copy the whole `sdcard/out/factorytest/` directory to its root, so the card has `factorytest/factorytest.sh` and `factorytest/dropbear`.
2. Card in, power on. Wait about 90 seconds — setup, wifi association, DHCP, and on the first run dropbear generating its host keys.
3. `ssh root@<hostname>` with the password you chose. If the name does not resolve, the card now has `pluto_setup.log` with the IP it got.

Note that the changes are not permanent, if you pull the SD card and reboot, the device will just come back with the original firmware and you can continue to use it as stock.