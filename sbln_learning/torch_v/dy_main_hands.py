import dynamic_handcard
import json
from flask import Blueprint,request,jsonify
sichuan = Blueprint("sichuan",__name__, url_prefix="/sichuan")
sichuan_hz = Blueprint("sichuan_hz", __name__, url_prefix="/sichuan_hz")

dy_tl = dynamic_handcard.DynamicHandCard()
@sichuan_hz.route("/get_last_by-rank", methods=["POST"])
def sichuan_hz_get_hand_by_rank():
    """
    输入为{
        "hands":[[], ....],
        "levels": [1,2,...]
    }
    Returns:
        [[],...]
    """
    input = request.get_json()
    res = dy_tl.gen_lastHands(input["hands"], input["level"])
    # print("hands:", res["hands"])
    # print("vault", res["vault"])
    return jsonify(res)
