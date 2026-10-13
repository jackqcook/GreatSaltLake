#!/bin/zsh
# Waits for run_hysplit_transfer.py to finish, then (only if it logged "done")
# runs Stage 4 episodes and the HYSPLIT-transport ensembles. Logs in logs/.
cd "$(dirname "$0")/.."
while pgrep -f run_hysplit_transfer.py > /dev/null; do sleep 60; done
if ! tail -1 logs/hysplit_transfer.log | grep -q " done$"; then
  echo "$(date) HYSPLIT year run did not finish cleanly; not continuing" >> logs/finish.log; exit 1
fi
echo "$(date) episodes start" >> logs/finish.log
.venv/bin/python scripts/run_episodes.py > logs/episodes.out 2>&1 || { echo "$(date) episodes failed" >> logs/finish.log; }
echo "$(date) ensembles start" >> logs/finish.log
.venv/bin/python scripts/run_model.py --ensemble 200 --transport hysplit --workers 8 > logs/run_model_hysplit.out 2>&1 \
  && echo "$(date) ensembles done" >> logs/finish.log || echo "$(date) ensembles failed" >> logs/finish.log
