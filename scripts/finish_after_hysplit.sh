#!/bin/zsh
# Waits for every run_hysplit_transfer.py (one or several date ranges) to finish,
# then, only if every chunk of the period is complete, runs Stage 4 episodes and
# the HYSPLIT-transport ensembles. Logs in logs/.
cd "$(dirname "$0")/.."
while pgrep -f run_hysplit_transfer.py > /dev/null; do sleep 60; done
if ! .venv/bin/python scripts/run_hysplit_transfer.py --check >> logs/finish.log 2>&1; then
  echo "$(date) HYSPLIT year run is incomplete; not continuing" >> logs/finish.log; exit 1
fi
echo "$(date) episodes start" >> logs/finish.log
.venv/bin/python scripts/run_episodes.py > logs/episodes.out 2>&1 || { echo "$(date) episodes failed" >> logs/finish.log; }
echo "$(date) ensembles start" >> logs/finish.log
.venv/bin/python scripts/run_model.py --ensemble 200 --transport hysplit --workers 8 > logs/run_model_hysplit.out 2>&1 \
  && echo "$(date) ensembles done" >> logs/finish.log || echo "$(date) ensembles failed" >> logs/finish.log
