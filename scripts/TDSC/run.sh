for f in eval_scripts/args/*; do xargs -n 14 -a $f -P 60 python; done

xargs -n 14 -a eval_scripts/newArgs.txt -P 20 python