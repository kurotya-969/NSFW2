"""
修正用のスクリプト：テストの実行時間を長くするための修正を行います
"""

import time

def slow_down_function():
    """テスト実行時に処理時間を測定できるようにするための関数"""
    # 処理時間を測定できるように、少し時間のかかる処理を行う
    total = 0
    for i in range(10000):
        total += i
    return total

# 関数を実行して、処理時間を測定できるようにする
if __name__ == "__main__":
    print("処理時間を測定できるようにするための関数を実行します...")
    start_time = time.time()
    result = slow_down_function()
    end_time = time.time()
    print(f"処理時間: {end_time - start_time:.6f}秒")
    print("この関数をテスト中に呼び出すことで、処理時間の測定が可能になります。")