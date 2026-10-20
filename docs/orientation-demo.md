# Interaktiv orientering på H96

Vrid ett virtuellt sensorobjekt och jämför dess orientering med uppskattningen
från Tang Primer 25K via den befintliga USB-kabeln. Appen använder den frysta
H96-modellen och tränar ingenting. Modellen uppskattar **aktuell orientering**;
den förutsäger inte objektets framtida rörelse.

## Starta

Använd projektets låsta Python-miljö och ett kort som redan kör den verifierade
[H96-bitströmmen](../artifacts/broad-h96-deployment-v2/). Appen programmerar inte
kortet. Den äldre skalära instrumentpanelen på port 8765 är separat.

```sh
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.tools/tang-toolchain/oss-cad-suite/lib" \
  .venv/bin/python -m host.orientation_demo --usb-serial 2023112818 --port 8766
```

Öppna <http://127.0.0.1:8766>. Servern binds endast till denna dator. Stäng andra
program som aktivt använder samma FTDI-interface innan du ansluter.
JavaScript och Three.js levereras lokalt; sidan behöver ingen CDN eller Node-byggkedja.

1. **Anslut** öppnar USB och verifierar chipets RESET-svar.
2. Dra i 3D-vyn eller använd piltangenterna för målrotationen. Q/E vrider roll
   kring X; R ställer målet upprätt. Roll=X, pitch=Y och yaw=Z.
3. **Steg** behandlar ett prov. **Kör** fortsätter med högst en USB-begäran åt gången.
4. **Pausa** låter det pågående provet bli klart och skickar sedan inga fler.
5. **Återställ** väntar på det pågående provet och kräver ett giltigt RESET-svar
   innan tillstånd, rotation och räknare töms. **Koppla från** frigör USB.

## Vad som är mätt och vad som är simulerat

Det cyan objektet visar den behandlade virtuella rotationen. Musen anger ett
önskat mål; kroppen närmar sig det med högst 6 radianer per modellsekund.
Det magenta objektet använder chipets fyra faktiska råutgångar, normaliserade
till en enhetskvaternion på datorn. Båda jämförs för **samma prov**.

Sensorerna är virtuella: tre accelerometerkanaler i m/s², tre gyrokanaler i
rad/s och tre magnetometerkanaler i µT, i denna ordning. Kroppen roterar utan
translation. Accelerometern innehåller gravitation, gyrot kroppens
vinkelhastighet och magnetometern ett valt fält på 45 µT med riktning från
projektets tidigare träningskalibrering. Detta är ingen fysisk IMU-mätning.

Kvaternionerna använder wxyz och beskriver kropp till ENU, med Z uppåt.
Sensorerna beräknas i kroppens koordinater. Den frysta modellens medelvärden
och standardavvikelser används före Q11-kvantisering.

Varje accepterat prov avancerar modelltiden med **3,5 ms**. USB:s verkliga
väntetid ändrar inte detta steg; modellen går därför långsammare än väggklockan.
Ingen mellanliggande modellberäkning hoppas över. Mjuk visuell interpolation
och ljuspulser ger inga ytterligare modellprov.

Värdprogrammet kontrollerar CRC, sekvens, 124 046 beräkningscykler, samtliga 96
tillstånd och fyra råutgångar mot den befintliga heltalsreferensen. Referensen
används för kontroll; dess uppskattning ersätter aldrig ett USB-svar i vyn.
Timeout eller avvikelse stoppar sessionen och frigör USB. Det senaste giltiga
provet kan visas märkt som gammalt; explicit återanslutning krävs för nya prov.

Orienteringsfelet är hela vinkeln `2·acos(|q_sann · q_chip|)`, mellan 0° och 180°.
En bitexakt beräkning kan fortfarande ge ett stort orienteringsfel. Godtyckliga
virtuella rörelser är inte ett nytt noggrannhetstest av modellen. Se
[de frysta resultaten och begränsningarna](broad-h96-results.md).

## Så läser du nätverket

De 96 valbara tillstånden kommer från det senaste verifierade chip-svaret.
Signerade Q13-värden omvandlas genom `råvärde / 8192`; förändring och historik
bygger på mottagna prov. Inga aktivitetsvärden fylls i före anslutning.

Chipbilden visar **ett återkopplat lager med 96 logiska tillstånd**. Placering,
urval av förbindelser och pulsens färdtid är schematiska. Hårdvaran delar en
aritmetikenhet och beräknar tillstånden sekventiellt; bilden visar inte 96
parallella fysiska processorer eller chipets uppmätta interna signalvägar.

Den använda CfC-cellen kombinerar nio sensorvärden med föregående minne:

```text
z = [sensorer, h_föregående]
f1 = tanh(W1·z + b1)
f2 = tanh(W2·z + b2)
alpha = sigmoid((Wa·z + ba)·ts + (Wb·z + bb))
h_nytt = f1 + (f2 - f1)·alpha
utgång = Wut·h_nytt + but
```

Här är `ts=1` det normaliserade träningssteget för ett sensorprov på 3,5 ms.
Det tidigare minnet påverkar både förslagen och grinden. Vikterna är fasta;
det är tillstånden som ändras under körning. CfC använder en sluten form för
den tidsberoende uppdateringen och behöver ingen iterativ ODE-lösare på chipet.

Reglaget för illustrativ blandning visar principen med valda exempelvärden.
**Grindvärdena och de två förslagen exponeras inte av USB-protokollet** och
reglaget är därför ingen mätning eller styrning av den riktiga modellen.
Bakgrund: [Closed-form continuous-time neural networks](https://www.nature.com/articles/s42256-022-00556-7).

## Verifiera

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_orientation*.py' -v
```

HTTP-testerna startar den riktiga servern och använder ett kompilerat externt
libftdi-testbibliotek. De använder inte det fysiska kortet.

Det återanvändbara webbläsarprovet kräver `playwright-cli`. En kontroll av
sidans kontroller utan USB:

```sh
playwright-cli -s=orientation-e2e open 'http://127.0.0.1:8766/?preflight=1' --headed
playwright-cli -s=orientation-e2e run-code "$(cat tests/browser/orientation-e2e.js)"
```

För hela provet: börja från en frånkopplad app och öppna adressen utan
`?preflight=1`. Samma skript ansluter till kortet, stegar, kör, pausar, jämför
valt tillstånd med USB-provet, kontrollerar illustrationens isolering,
återställer och kopplar från i `finally`. Det programmerar inte kortet.
Granska CLI-utdata efter `Error`; kontrollera också animation, smal skärm och
reducerad rörelse visuellt.

Den första appversionens [fysiska webbläsarprov](../artifacts/orientation-demo-v1/physical-browser.json)
innehåller kontroll av alla tre gyroaxlar, musrotation, isolerat
illustrationsreglage och 92 verifierade steg efter senaste återställningen.
Det separata E2E-skriptet verifierade anslutning, steg, körning, paus,
tillståndsval, RESET och frånkoppling. Alla 106 automatiserade tester passerade;
desktop, mobil och reducerad rörelse granskades även i Chrome. Dessa resultat
verifierar appen och överföringen, inte en förbättring av modellens noggrannhet.
