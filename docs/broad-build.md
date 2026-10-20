# Bygga BROAD för Tang Primer 25K

`scripts/build_broad.py` bygger en verifierad heltalsexport och kräver
laglig placering/routing, minst 50 MHz och ett icke-tomt packat bitflöde.
Byggkedjan har klarat både H3 och hela H96-designen med UART.

```sh
PATH="$PWD/.tools/tang-toolchain/oss-cad-suite/bin:$PATH" .venv/bin/python scripts/build_broad.py --export-dir artifacts/broad-integer-v1/export --build-dir build/broad --seed 3
```

Utmappar måste vara nya eller tomma och ligga utanför den frysta exporten.
Körningen bevarar syntes-, placerings-, timing- och packningsloggar. Den
skapar `broad.fs` och `report.json` först efter respektive kontroll.
Ingen kortprogrammering ingår i byggkommandot.

Rapporten binder modellens export, koefficienter, 1 024-ords viktbanker,
RTL, constraints, byggskript och verktyg med hashar. Den kontrollerar
separata fysiskt mappade ursprung för vikt-, tanh- och sigmoidminnen.
Uppnådd timing och resursutnyttjande kommer från routingverktyget.

För explicita verktygsinställningar:

```sh
PATH="$PWD/.tools/tang-toolchain/oss-cad-suite/bin:$PATH" .venv/bin/python scripts/build_broad.py --export-dir artifacts/broad-integer-v1/export --build-dir build/broad-explicit --placer heap --seed 3 --heap-beta 0.9 --stage-timeout 1200
```

## H96-status

Den bitexakta H96-implementationen har laglig placering/routing och ett packat
bitflöde. [Byggrapporten](../artifacts/broad-h96-deployment-v2/build/report.json)
visar **50,5561 MHz Fmax** mot kravet 50 MHz. Kortets klocka är
fortfarande **50 MHz**; Fmax är routingverktygets timingresultat.

Designen använder **11 361 av 23 040 LUT4**, **4 218 register** och
**48 av 56 blockminnen**: 41 för vikter, fyra för tanh och tre för sigmoid.
Byggskriptet väljer uttryckligen `router2`. Explicita tillståndsskrivningar
och tidigare registrering av `gate_sum` behåller talformat och cykelschema;
[10 000 sammanhängande RTL-steg](broad-rtl.md) matchar heltalsreferensen exakt.
Tidigare placeringsfel, tidsgränser och timingfelet före `gate_sum`-ändringen
finns kvar i [byggarkivet](../artifacts/broad-capacity-v1/history.json).
Den nya avrundningsimplementationens misslyckade försök och det reproducerade
seed-3-bygget finns i [driftsättningshistoriken](../artifacts/broad-h96-deployment-v2/history.json).
Builderns standard är fortfarande seed 1; H96-kommandona ovan väljer uttryckligen
seed 3. `make broad-build` använder standarden och återskapar inte seed-3-bygget.

[Verifieringen](../artifacts/broad-h96-deployment-v2/verification.json) omfattar
94 unika tester i två separata körningar: kapacitetstestet (79,390 s) och
övriga 93 tester (229,062 s). Compileall, strikt Verilator-lint för H3/H96
och oberoende granskning passerade också.

[Sluttestet och fysisk USB-verifiering](broad-h96-results.md) redovisas separat.
Den nya implementationen matchar 15 000 fysiska steg exakt, inklusive 1 000
rörelseprov. Det tidigare byggets fysiska tillståndsbitsfel finns bevarat;
den nya implementationen ändrar inte modellen eller dess uppmätta accuracy.
Den äldre skalära designens 600-sekunders USB-uthållighetskrav är fortfarande öppet.
