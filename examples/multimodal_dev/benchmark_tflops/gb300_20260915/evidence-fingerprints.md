# Evidence fingerprints — working draft

These SHA-256 fingerprints identify original retained artifacts. They do not make private artifacts downloadable, certify a corrected rate, or replace consumed-boundary proof. Identifiers are portable; no host paths or raw data are included.

| Result identifier | Original raw-log SHA-256 | Original accepted-result SHA-256 |
|---|---|---|
|EXP-PR131-ENCODER-CP1-GB300| `16116c3b7f2aaa2e61f3eca517b60b85c88ca12682a835ffbefaa3b71a2146ec` | `077f141658bc7a7c851933e4c3672bf3b46dddafa9312b033fdd1e94d657570f` |
|EXP-PR131-ENCODER-CP1-PAIR749724-GB300| `7623be127c37f652c91da3764e30b3a69d01bbf2f616fce4fbfa5ec75b90a7c5` | `a8dfd615c8022a0ca942f95b1c397b0c46acc4bf466f81644ce31dfcf5c6631f` |
|EXP-PR131-ENCODER-CP2-PAIR749724-GB300| `1213cf8af8de036bcb245df1750666e61f7d93145cc6045af220f60e18b51d7b` | `bdb340a6f4b7e5f420b28445903cc6b635336276f7f2d6d290d9868bdf12297e` |
|EXP-PR131-IMAGE1X-CP1-GB300| `23fe3e0599dc1b1cf6d64e7df553b7563ae6804be6f3ab6e8c597fa83b8a19bc` | `6bb7901cacd24dd90bd5e3cec8490e5eb9c2a8dbbb9a138cdc53bbfde4ddb150` |
|EXP-PR131-IMAGE1X-CP2-GB300| `0cc3aa0a8bd2e8fc03321c2fd3102bbb7607a065c37aa812d3152da1fe7a1073` | `52671edc484b91411df4f27a0445560e7b1411c46b5d300a7d1b522909c76399` |
|EXP-PR131-IMAGE2X-CP1-GB300| `a3714927255ae5f10613b732b9bd20ca094994496dec9fec256886e5b15f3103` | `8a80296a41c6644ff5164167a56866f4a5d13d057b1eee28a9c0dc28735dfc0c` |
|EXP-PR131-IMAGE2X-CP2-GB300| `7f9fa4b47e765d75c2c0ca7db3e6b671fa151ac14e63fa3634832247087a4312` | `526eb5a4410a3840ab9eb8e9bb8fe717a3e91c6c0f529534568accc8479112a7` |
|EXP-PR131-IMAGE4X-CP1-GB300| `4330b30b003f237b18aff3dd421c332cdc18f01da10b67bb6f2732af8fdb19db` | `baaaf0b19553f5a927364c3bb0fb53e87c4e0b0c5ce5007be4a4a9b7688f773b` |
|EXP-PR131-IMAGE4X-CP2-GB300| `4f49dfbd1cf2e1dd731a2e0edf570c2f61517b37a9849437270d1d8ceb311001` | `2ce177fe328f5f242c463abb3488c8fa164c2a815d73dfbfc0f77e9ccba91e07` |
|EXP-PR131-IMAGE8X-CP1-GB300| `e40cc1352ce383d586c385912dc099035a24fb353d7c0f2409f8255c9f141aac` | `4e9ec93ef394b0503fe30a5caac90d8c402cde51d82aaad53e42913897b5fa80` |
|EXP-PR131-IMAGE8X-CP2-GB300| `7ed4b348e63647d2b4cf2d01482c5f98b59b4a77c37cc9818539cfbf499e760c` | `34eb1a6394404b036e7ccd1849c87f1a78f19660bdda336e8b04f14583e172bd` |
|EXP-PR131-MANTIS16-752807-20-GB300-legacy-window| `8a8a11f9a60051005a1ee827e0c48542075e270f2bc8bee2643dc27c06c88ef8` | `1e48d569608b1828457e9f0f3d1b459ef1313cf02b1ee286e14602b085ea517b` |
|EXP-PR131-TFLOPS16-SEQ4096-GBS256| `910bee2a5c815cec36a54805acb9431a12b69bf6c017b3f061b2b550f82d9176` | `c7836b501d98c48cf7c037efaf51f640da9679c52d720afd9eff85b2321f6539` |
|EXP-PR7-FOURWAY-751915-pr7_baseline| `f7aa6900b9a61fda6f3a1a4aaf6795981200b6ef26600e395924253e5360a6d0` | `113f94469377d596d70173979cca2e52062cd14534c9481f647f5105d26cfe2e` |
|EXP-PR7-FOURWAY-752159-pr131_latest| `ed914d8c4c743166515a96bcf96ca510c54a326375d91c81d9a37b5e0707eb4b` | `89566c140272a4977765cd483d60340c38207505d9e115b1e9a73ca515b983b7` |
|EXP-PR7-FOURWAY-752159-pr7_baseline| `1082aa68c58b7bb94be7d99e2875a80af2e08531efb358041566a2f45d8fed66` | `0856fd1a73c51eca969c149e8ced8624b536186937c1b1226fdce28c4f55890a` |
|EXP-PR7-FOURWAY-752159-pr7_mdp| `ce5204c67dc82b7c4c5d0296131b8bdb9282251c96b4ced3448c3b41a4206726` | `8c7dd29d5cfd9edbf34dabdb7169b8db24d23648b849ddaa45969669e560e880` |
|EXP-PR7-FOURWAY-752159-pr7_packing| `227e65333842503b28823e3ea836cb75b1853e85e05c0875de7119296cff00b2` | `dd03ff43f27204797384e299d46e64b876667fdaf106c13da86fc51599c38bba` |

Prepared dataset, tokenizer revision, and preflight fingerprints are listed in [data-and-configs.md](data-and-configs.md). Recorded early argument values and final resolved model values must remain distinct. This manifest was generated from the owned inventory and accepted JSON bytes; no external artifacts were fetched.

## Historical log retrieval limits

A bounded metadata-only audit covered the 231 catalog artifacts. For 189 historical references, exact filenames were absent at all three checked, known owned locations (567 unique checks, all `ENOENT`). Another 38 previously located owned-campaign references were deliberately not rechecked; four artifacts have no log reference and are not counted as proved file absence. No new candidate logs were found, and no permission failures occurred in these checks.

This is a retrieval limitation at those three locations, not evidence that no backup exists elsewhere. No broad filesystem or peer-worktree search was performed. Original timing rows and resolved runtime arguments remain necessary: recovered source definitions alone cannot establish corrected historical rates. The private coverage receipt has SHA-256 `c8e3c73ceff9cfe341801963020051e7c8a441b9cdd9b79be84f74a7b9721fa6`; its host paths and raw records are not published.
