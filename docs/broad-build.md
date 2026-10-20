# Bygga BROAD för Tang Primer 25K

`scripts/build_broad.py` bygger en verifierad heltalsexport och kräver
laglig placering/routing, minst 50 MHz och ett icke-tomt packat bitflöde.
Byggkedjan har klarat ett riktigt H3-prov med åtta verifierade blockminnen.
91 tester och compileall passerade; oberoende granskning passerade också.

```sh
make broad-build BROAD_EXPORT_DIR=artifacts/broad-integer-v1/export BROAD_BUILD_DIR=build/broad
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
PATH="$PWD/.tools/tang-toolchain/oss-cad-suite/bin:$PATH" .venv/bin/python scripts/build_broad.py --export-dir artifacts/broad-integer-v1/export --build-dir build/broad-explicit --placer heap --seed 1 --heap-beta 0.9 --stage-timeout 600
```

## H96-status

H96 är ännu inte godkänd för fysisk laddning. Den första kompletta syntesen
gav 22 361 LUT4, 4 689 register och 48 blockminnen. Två heap-försök hittade
ingen laglig placering; SA-placeraren avslutades med segmenteringsfel.

Diagnosen pekar på syntesens översättning av `new_hidden`: hel-array-reset
och dynamisk skrivning skapade en uppblåst muxstruktur och 128 ord trots
96 logiska tillstånd. Nästa begränsade ändring gör varje registers skrivning
explicit, med samma två registerbanker, gränssnitt, talformat och cykelschema.
Ny faktisk H96-passning och sammanhängande RTL-jämförelse krävs efteråt.
