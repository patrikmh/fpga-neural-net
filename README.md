# FPGA Neural Net – Tang Primer 25K

Ett stegvis projekt för signalprediktion med ett litet CfC-nät på Tang Primer 25K.
Modellen tränas på datorn. FPGA:n ska köra inferens med fasta vikter.

## Status

Provet för UART, signerad multiplikation, ROM och RAM är simulerat och byggt.
FPGA:n är identifierad genom JTAG som Gowin GW5A-25, ID `0x1281b`, och provet
har laddats i SRAM. Det ordinarie USB-kommandot har verifierat 1 000 av
1 000 riktiga svar. CfC-modellen är tränad och testad på datorn; nätets
FPGA-inferens återstår.

PyTorch 2.14.1, ncps 1.0.1 och NumPy 2.4.6 är versionslåsta.
Den verifierade cellen har tillsammans med
utgångslagret 329 parametrar. På låsta testdata är modellens MSE 0,00493,
jämfört med 0,04941 för senaste värdet och 0,01176 för linjär extrapolation.
Det motsvarar 90 % respektive 58 % lägre fel. Se
[testresultatet](reports/float-heldout-2026-10-19.json).

Provbygget klarar kravet 50 MHz (rapporterad gräns 82,37 MHz) och använder 2 370
LUT4, 308 register, 124 ALU och tre blockminnen. Flash har inte skrivits.
Den vanliga serieporten fungerade inte på denna Mac; libftdi via samma USB-kabel
fungerar med baudbegäran 460800 mot FPGA:ns 115200 baud. Medeltiden för en
beräkning inklusive USB tur och retur är 6,00 ms (p95 6,04 ms).

Den sekventiella CfC-kärnan matchar alla åtta tillstånd och utvärdet exakt under
10 000 sammanhängande steg i RTL-simulering. Beräkningen tar 1 052 klockcykler,
vilket motsvarar 21,04 µs vid simulatorns 50 MHz. Detta är ännu ingen
timingrapport eller fysisk körning av neuronnätet.

Se [genomförandeplanen](docs/implementation-plan.md),
[det fysiska provet](reports/probe-summary-2026-10-19.json) och
[RTL-resultatet](reports/rtl-summary-2026-10-19.json).
Återstår: resetkommandot i CfC-protokollet, CfC-byggets timingkontroll,
verklig nätinferens och den längre demonstrationen på kortet.

Ett prov med signerad 16x16-multiplikation visar att installerad Yosys använder
vanlig logik och inga DSP-block för `-family gw5a`. Första versionen använder
denna lösning efter godkännande. Resursåtgång och timing har verifierats före laddning.

## Kör lokalt

Python 3.11 används för projektet. Skapa den låsta miljön före Make-kommandona:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked
```

Själva miljökontrollen kan fortfarande köras direkt med Python-standardbiblioteket.
Verifiering över seriell port använder den låsta dependency-versionen pyserial 3.5.

```sh
make check
make doctor
python3 scripts/check_env.py --json
```

`make test` kör diagnostiktester, verklig UART-simulering, ett FPGA-bygge och
värdprogrammets seriella tester med externa testmotparter. Det programmerar
ingen FPGA. `make doctor` returnerar felkod om något verktyg saknas
eller dess kontrollkommando misslyckas.

Miljökontrollen kan köras utan virtuell miljö. Tester och modellkommandon
använder den låsta miljön med Python 3.11. Testerna skapar egna tillfälliga
modellfiler och skriver inga FPGA-konfigurationer till hårdvara.

## Verktyg som behövs inför kortprovet

- `uv`
- `yosys`
- `nextpnr-himbaechel`
- `gowin_pack` från Apicula
- `openFPGALoader`
- `iverilog` och `vvp` för RTL-simulering

En möjlig distribution är [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build),
som innehåller verktyg för syntes, placering och simulering. Välj en macOS-version
som passar datorns arkitektur, följ distributionens installationsanvisning och
kör miljökontrollen igen. Kontrollera även om `openFPGALoader` ingår i den valda
distributionen. Det provade bygget använder utgåvan
[2026-10-19 för macOS ARM64](https://github.com/YosysHQ/oss-cad-suite-build/releases/tag/2026-10-19).
Inga systemverktyg installeras automatiskt av detta projekt.
En lokal verktygsinstallation kan placeras under `.tools/` i projektet.
Standardplatsen är `.tools/tang-toolchain/oss-cad-suite`. Make-kommandona använder
den automatiskt. För en annan installation anger du `TOOLCHAIN_DIR=/sökväg/till/oss-cad-suite`.
Verktygsinstallationer, virtuella miljöer och byggfiler ingår inte i repot.

## Reproducera provbygget

```sh
make sim-probe
make probe-build
```

`build/probe/report.json` innehåller källfilernas och bitströmmens SHA-256,
verktygsidentiteter, faktisk minnesmappning, resursåtgång och timing.
Det fysiska resultatet från det ordinarie kommandot finns i
[kortprovet](reports/probe-summary-2026-10-19.json).

När just provbitströmmen körs på kortet:

```sh
make verify-usb USB_SERIAL=DITT_USB_SERIENUMMER COUNT=1000
```

Kommandot väljer endast detta USB-serienummer och kanal B. Det programmerar
inte kortet. På denna Mac sätter Make den lokala bibliotekssökvägen för libftdi.
Testerna kontrollerar även timeout, felaktiga svar och att USB-resurser frigörs
vid fel. Efter ett strömavbrott behöver provet laddas i SRAM igen.

Miljödiagnostiken listar endast serieportskandidater. Den öppnar inga portar
och gör inga JTAG-anrop. Flashprogrammering ingår inte.

## Modell och reproducerbarhet

Den tränade modellen, 329 fasta vikter, aktiveringstabeller och 10 000
referenssteg finns i [artifacts/cfc8-q13-v1](artifacts/cfc8-q13-v1).
Viktfilerna innehåller endast den syntetiskt tränade modellen. Manifesten
binder koefficienter, tabeller, kodversioner och numeriska regler med SHA-256.

Förbered standardplatsen för RTL-kärnans ROM-filer i en ny klon:

```sh
mkdir -p build
cp -R artifacts/cfc8-q13-v1 build/integer-v1
```

För att träna om exakt samma modell från början, välj en ny utmatningskatalog:

```sh
.venv/bin/python -m model.train --output-dir build/retrained --epochs 500 --seed 20261019 --train-trajectories 32 --validation-trajectories 8 --length 256
.venv/bin/python -m model.export_integer --checkpoint build/retrained/checkpoint.pt --output-dir build/retrained-integer
.venv/bin/python -m model.evaluate_integer --export-dir build/retrained-integer --output-dir build/retrained-validation
```

Tränings- och valideringssekvenser skapas oberoende. Testdata används först
efter att modellen och utvärderingsspecifikationen har låsts. Den sparade
[testspecifikationen](reports/float-test-spec-2026-10-19.json) kräver den
ursprungliga checkpointens filhash; en ny Torch-arkivfil kan få annan filhash
även när modellens tensorer är identiska.


## Så gör modellen en förutsägelse

Varje nytt värde `x` kombineras med nätets åtta tidigare tillstånd, `h`.
Två grenar föreslår nya tillstånd och en sigmoidgrind blandar dem:

```text
z = [x, h_gammalt]
f1 = tanh(W1*z + b1)
f2 = tanh(W2*z + b2)
g = sigmoid(Wa*z + ba + Wb*z + bb)   # fast normaliserat tidssteg = 1
h_nytt = f1 + (f2 - f1)*g
y = Wut*h_nytt + but
```

Alla neuroner läser hela det gamla tillståndet. Utgången använder sedan hela
nytillståndet och förutsäger nästa sampel, inte det redan observerade värdet.
Tillståndet nollställs mellan oberoende signalsekvenser. Medelvärde och skala
beräknas enbart från träningsdata.

Den låsta heltalsreferensen använder signerade 16-bitars tal med 13 bråkbitar:
`1,0` representeras som `8192`, med intervallet `[-4, 4)`. Produkter ackumuleras
i 48 bitar. Avrundning vid halva steg går från noll åt båda håll. Aktiveringarna
använder två tabeller med 4096 värden vardera. Vikter, tabeller, avrundning och
operationsordning finns i `build/integer-v1/manifest.json`.

På låsta testdata ökade heltalsmodellens MSE med 0,061 % jämfört med
flyttalsmodellen, utan klippning. [Heltalstestet](reports/integer-heldout-2026-10-19.json)
är körning på datorn. Bitexakt RTL-simulering är också verifierad; fysisk
FPGA-inferens med nätet återstår.
