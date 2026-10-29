#!/usr/bin/env bash
# Rebuild CmdStan's values in tests/reference/<model>.json.
#
# Each reference file holds a model's data and test point, which this script
# keeps, and what CmdStan computes from them, which it rewrites:
#   log_prob  the log density, constants included, and its gradient at the
#             test point, with the Jacobian adjustment;
#   draw      the output names and one draw's values, in Stan's CSV order.
#
# The tests compare against these values, so they never need CmdStan. Run
# this by hand after changing a model in tests/models/:
#
#   tests/reference/update.sh            # every model
#   tests/reference/update.sh mvn mix    # just these
#
# Set CMDSTAN to a CmdStan 2.37 or newer, whose Stan Math has the simplex
# transform that the tests build against. CmdStan's builds of the models are
# kept in build/reference/ and rebuilt when a model changes.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODELS="$ROOT/tests/models"
WORK="$ROOT/build/reference"
CMDSTAN=${CMDSTAN:-$HOME/cmdstan}

version="$CMDSTAN/stan/lib/stan_math/stan/math/version.hpp"
if [ ! -f "$version" ]; then
  echo "no CmdStan at CMDSTAN=$CMDSTAN" >&2
  exit 2
fi
major=$(sed -n 's/#define STAN_MATH_MAJOR *//p' "$version")
minor=$(sed -n 's/#define STAN_MATH_MINOR *//p' "$version")
# Stan Math 5.1 changed the simplex transform to the one stancpp uses.
if [ "$((major * 100 + minor))" -lt 501 ]; then
  echo "CmdStan's Stan Math is $major.$minor; 5.1 or newer is needed" >&2
  exit 2
fi

if [ $# -gt 0 ]; then
  names=("$@")
else
  names=()
  for stan in "$MODELS"/*.stan; do
    names+=("$(basename "$stan" .stan)")
  done
fi

mkdir -p "$WORK"
for name in "${names[@]}"; do
  ref="$HERE/$name.json"
  exe="$WORK/$name"
  echo "== $name"

  # Copy the model only when it changed, so make rebuilds only then.
  cmp -s "$MODELS/$name.stan" "$exe.stan" || cp "$MODELS/$name.stan" "$exe.stan"
  (cd "$CMDSTAN" && make "$exe" >/dev/null)

  python3 - "$ref" "$exe" <<'PY'
import json, sys
ref, exe = sys.argv[1], sys.argv[2]
r = json.load(open(ref))
json.dump(r["data"], open(exe + ".data.json", "w"))
json.dump({"params_r": r["theta"]}, open(exe + ".theta.json", "w"))
PY

  "$exe" log_prob unconstrained_params="$exe.theta.json" jacobian=1 \
    data file="$exe.data.json" output file="$exe.log_prob.csv" \
    sig_figs=17 >/dev/null
  # One draw; the first iterations' rejections are expected and silenced.
  "$exe" sample num_warmup=1 num_samples=1 random seed=1 \
    data file="$exe.data.json" output file="$exe.sample.csv" \
    sig_figs=17 >/dev/null 2>&1

  python3 - "$ref" "$exe" <<'PY'
import json, sys
ref, exe = sys.argv[1], sys.argv[2]

def rows(path):
    return [l.strip().split(",") for l in open(path)
            if not l.startswith("#") and l.strip()]

r = json.load(open(ref))
header, values = rows(exe + ".log_prob.csv")
r["log_prob"] = {"lp": float(values[0]),
                 "gradient": [float(v) for v in values[1:]]}
header, *draws = rows(exe + ".sample.csv")
keep = [i for i, name in enumerate(header) if not name.endswith("__")]
r["draw"] = {"names": [header[i] for i in keep],
             "values": [float(draws[-1][i]) for i in keep]}

with open(ref, "w") as f:
    f.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}"
                               for k, v in r.items()) + "\n}\n")
PY
done
