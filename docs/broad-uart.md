# BROAD: UART och USB-verifierare

Den nya transportkedjan har verifierats genom verkliga UART-signaler i
RTL-simulering. RESET och två återkopplade steg körde den riktiga kärnan.
De fångade svaren matades genom ett separat native FTDI-testbibliotek till
verifierarens riktiga CLI. Alla tillstånd, fyra råutgångar, CRC,
sekvensnummer och uppmätta cykler stämde. Detta var inget fysiskt kortprov.

Alla 87 tester och compileall passerade efter implementationen, liksom strikt
Verilator-lint för H3/H96. Efteråt passerade felinjicering för skadad
request-CRC, felaktiga svar, sekvensfel och ofullständiga native läsningar.
Produktionskällorna ändrades inte. En felaktig request-CRC gav inget svar och
ändrade inte nätets tillstånd; efterföljande giltiga steg matchade referensen.

## Protokoll

Alla flervärdesfält är little-endian. CRC är CRC-16/CCITT-FALSE:
polynom `0x1021`, start `0xffff`, ingen reflektion, xorout noll.
Kontrollvektorn `123456789` ger `0x29b1`. CRC-ordet sänds little-endian och
täcker alla föregående byte i paketet.

| Request, 24 byte | Innehåll |
|---|---|
| Byte 0 | Magic `0xa6` |
| Byte 1 | STEP = 1, RESET = 2 |
| Byte 2–3 | Sekvensnummer |
| Byte 4–21 | Nio signerade Q11-indata |
| Byte 22–23 | CRC16 |

Svaret innehåller magic, status, sekvensnummer, H signerade Q13-tillstånd,
fyra signerade Q13-råutgångar, 32-bitars cykelantal och CRC16. Storleken är
`18 + 2H` byte, alltså 210 byte för H96. RESET svarar med nollställda
tillstånd, råutgångar och cykler. Värden normaliseras inte vid RESET.

Transporten skickar en request och inväntar hela svaret. Fel CRC eller
okänt kommando ignoreras av RTL. Värdkoden avbryter vid ofullständiga svar,
fel status, sekvens, tillstånd, utgångar eller cykler. Ett misslyckat prov
publicerar ingen lyckad resultatrapport.

## Körning

Efter [RTL-studien](broad-rtl.md) kan USB-vektorerna återskapas med:

```sh
.venv/bin/python artifacts/broad-uart-v1/package-vectors.py --export-dir artifacts/broad-integer-v1/export --rtl-dir build/broad-rtl-v1 --output-dir build/broad-usb-vectors-v1
```

Utmappar måste vara nya. Paketets indata är de första 10 000 sammanhängande
valideringsproven, bundna till export, checkpoint och RTL-rapport med hashar.
Verifieraren kräver explicit USB-serienummer och uppmätt cykelantal.
Exempel efter att motsvarande SRAM-bygge har laddats:

```sh
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.tools/tang-toolchain/oss-cad-suite/lib" .venv/bin/python -m host.verify_broad --usb-serial 2023112818 --export-dir artifacts/broad-integer-v1/export --vectors-dir build/broad-usb-vectors-v1 --count 1000 --expected-cycles 124046 --output build/broad-usb-report.json
```

UART går fysiskt i 115200 baud, med 434 klockor per bit vid 50 MHz.
Den verifierade Mac-konfigurationen begär 460800 genom libftdi.
Kärnans beräkningstid och USB tur och retur redovisas separat; hela
USB-varvet inkluderar beräkningen och är inte samma sak som kärnans kapacitet.

Faktisk passning, timing och fysisk verifiering av H96 återstår. Den första
placeringskörningen hittade ingen laglig placering vid cirka 97 % LUT-utnyttjande;
andra placeringsinställningar provas med samma modell och RTL.
