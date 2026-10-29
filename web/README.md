# Web experiment: recognising AI avatars (up to 3 seconds vs unrestricted viewing)

Node.js + Express, no database: finished sessions are appended to `data/participant_trials.csv`.
The file format is compatible with the project's `src/process_data.py` (columns `participant_id, condition, stimulus_id, is_ai, user_choice, response_time_ms`).

## Running

```bash
npm install
npm run demo-stimuli     # only for a trial run: 24+24 placeholder pictures (not faces)
npm start                # http://localhost:3000
npm test                 # end-to-end check of the server and the data format
```

To collect data, **delete `stimuli/*/demo_*`** and put the real images into `stimuli/real/` and `stimuli/ai/`
(`.jpg .jpeg .png .webp`). At least `TRIALS_PER_BLOCK` images per class are needed (20 by default; more is better,
because then participants get different sets). A subfolder inside `stimuli/ai/` (for example `stimuli/ai/stylegan3/`) is written to the `generator` column.
While the folders contain files with `demo` in their name, a red banner is shown on the page.

Stimuli can be prepared with one command (requires `pip install pillow`):

```bash
python scripts/prepare_stimuli.py --real path/to/real --ai path/to/generated --generator stylegan3
```

The script takes the same number of images from each class (real ones are chosen randomly, seed 42), converts both classes to the same size
and format (512×512 JPEG, quality 92 by default, metadata stripped), places them into `stimuli/real` and `stimuli/ai/<generator>`,
removes the demo placeholders and writes `stimuli/manifest.csv` (source file → new file). Before running it, remove images with obvious artefacts
from the input folders. Images are not committed (`.gitignore`); `manifest.csv` may be committed.

Settings are given through environment variables: `PORT`, `TRIALS_PER_BLOCK` (even, 20 by default), `VIEW_LIMIT_MS` (3000), `DATA_DIR`, `STIM_DIR`.

## How the experiment works

- Consent (18+, anonymous) → field of study (IT / humanities / other) → two blocks → saving.
- Every participant takes both conditions. The block order alternates with the arrival of participants: `3s_first` and `unrestricted_first` (AB/BA).
- Each block is 1:1 real/AI, the stimuli of the two blocks do not overlap, and the order of trials is random.
- The side of the answer buttons ("Real photo" on the left or on the right) is random per participant: column `button_layout`.
- The truth (`is_ai`) is stored on the server only, and images are served under hashed identifiers: the class cannot be learned from file names.
- **3s_limit:** the picture and the buttons appear together. The participant can answer immediately, and then the picture disappears at the moment of the answer.
  If there is no answer within 3 s, the picture disappears by itself, the buttons stay, and the answer is accepted afterwards.
  **unrestricted:** the picture stays until the participant answers.
- **Definition of RT** (put it in the report): in both conditions `response_time_ms` is measured from the moment the picture and the buttons are painted,
  so the values are comparable between conditions. `exposure_ms` is how long the picture was on screen:
  in `unrestricted` it equals RT; in `3s_limit` it equals RT for an early answer and ~3000–3020 ms if the picture was hidden by the timer
  (values clearly above the limit mean a timer failure, for example because of a minimised tab).
- `focus_lost = 1` if the tab lost focus during the trial (a candidate for exclusion).

## Columns of `data/participant_trials.csv`

| Column | Meaning |
|---|---|
| `participant_id` | random id, no personal data |
| `condition` | `3s_limit` or `unrestricted` |
| `block_order`, `block_index`, `trial_index` | block order, block number, trial number within the block |
| `stimulus_id`, `is_ai`, `generator` | stimulus, truth (1 = AI), source |
| `user_choice` | answer: 1 = "AI-generated", 0 = "Real photo" |
| `response_time_ms`, `exposure_ms`, `focus_lost` | response time, exposure time, loss of focus |
| `button_layout`, `major`, `device_type` | button layout, field of study, `mobile`/`pc` |
| `submitted_at` | time of saving (UTC) |

The positive class is AI, as in your pipeline: FPR = real judged as AI, FNR = AI judged as real.

## Connecting to the project

The folder already lives in the repository as `web/`. When data collection is finished, run `python src/import_web_data.py` from the repository root
(it validates the format and copies the file to `data/real/participant_trials.csv`), then set this path in `dataset.human_trials_path`
in `config.yaml`. Details and the validity thresholds for response time are in the main `README.md`, section "Web experiment and connecting real data".

## Making it available to participants

Recruiting over the internet needs a public address: `ngrok http 3000` or `cloudflared tunnel` temporarily; Render, Railway or Fly.io (`npm start`) for the long term.
The file system on free hosting is often temporary, so download `participant_trials.csv` after every day of collection or attach a persistent volume.

## Limitations

- Only fully finished sessions are saved. If the server restarts in the middle of a session, that session is lost.
- The block order alternates by the number of started sessions, so with many unfinished sessions the AB/BA balance shifts slightly. Check `block_order` at the end.
- Timer accuracy is limited by the screen refresh rate (one frame ≈ 16 ms). This is acceptable for a 3-second exposure, but the devices should be stated in the methodology.
- The consent text is generic: align it with your university's requirements and add the researchers' contact.
