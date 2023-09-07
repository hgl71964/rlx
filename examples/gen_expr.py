import os
import random

from absl import app
from absl import flags

import numpy as np

from rlx.extern.expr.expr_utils import get_lang, save_expr, cnt_op
from rlx.extern.expr.math_def import verify

FLAGS = flags.FLAGS
flags.DEFINE_string("lang", "MATH", "")
flags.DEFINE_integer("depth_lim", 5, "depth limit of expression tree")
flags.DEFINE_integer("l", 0, "whether to log")

flags.DEFINE_integer("seed", 42, "")
flags.DEFINE_string("default_out_path", "data", "output dir")


def main(_):
    # set random seeds for reproducability
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # file name
    tmp_foder = f"{FLAGS.default_out_path}/rlx/inputs/"
    tmp_file = tmp_foder + f"{FLAGS.lang}-{FLAGS.depth_lim}-{FLAGS.seed}.pkl"

    if not os.path.exists(tmp_foder):
        os.makedirs(tmp_foder)

    # ==========================
    lang = get_lang(FLAGS.lang)()
    expr = lang.gen_expr(depth_limit=FLAGS.depth_lim)

    print("expr::")
    print(expr)

    # print("verify::")
    # vv = verify(expr)
    # print(f"verified value: {vv}")

    cnt = cnt_op(expr)
    print(f"seed: {FLAGS.seed}")
    print("num of ops: ", cnt)

    # save is OK
    l = bool(FLAGS.l)
    if l and cnt > 400:
        save_expr(expr, tmp_file)
        print(f"Save {FLAGS.seed} OK")


if __name__ == "__main__":
    app.run(main)
