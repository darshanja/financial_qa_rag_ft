import re

def normalize_answer(ans: str) -> str:
    # Remove commas in numbers like 57,094
    if re.match(r"^\d{1,3}(,\d{3})+$", ans):
        ans = ans.replace(",", "")
        try:
            val = int(ans)
            # Convert to billions if > 10^9
            if val > 1e9:
                return f"${val/1e9:.2f} billion"
            # Convert to millions if > 10^6
            elif val > 1e6:
                return f"${val/1e6:.2f} million"
            else:
                return f"${val}"
        except:
            return ans
    return ans