"""
Code taken from the repo: github.com/tamassimonds/CustomTinyZero
In the file verl/utils/reward_score/utils/llm_judge_utils.py.
Just slightly modified to add status of the rewards.
"""
import re
from enum import Enum, auto

class JudgeStatus(Enum):
    OK = auto()
    NO_OK = auto()
    NO_SCORE_IN_TAGS = auto()
    NO_AGENT_RESPONSE = auto()
    NO_INP_IN_TAGS = auto()
    ERROR = auto()

def extract_judge_score(response_str: str, method: str = 'strict'):
    """
    Extracts the candidate integration solution from the provided solution string.
    Also filters out any candidate that directly contains an integration command.
    """
    if not response_str or not isinstance(response_str, str):
        return 0, JudgeStatus.NO_AGENT_RESPONSE
        
    assert method in ['strict', 'flexible'], "Method must be 'strict' or 'flexible'"
    return_score = None
    if method == 'strict':
        try:
            matches = re.findall(r"<JUDGE_SCORE>(.*?)</JUDGE_SCORE>", response_str, re.IGNORECASE | re.DOTALL)
            return_score = matches[-1].strip() if matches else None
        except Exception:
            return 0, JudgeStatus.NO_SCORE_IN_TAGS
    else:
        return_score = response_str.strip()
    
    try:
        return_score = int(return_score)
    except Exception:
        return 0, JudgeStatus.ERROR
    else:
        if return_score == 1:
            return 1, JudgeStatus.OK
        elif return_score == 0:
            return 0, JudgeStatus.NO_OK
        else:
            return 0, JudgeStatus.ERROR

    # return return_score, JudgeStatus.OK