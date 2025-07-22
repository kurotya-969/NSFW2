"""
Test script for prompt generator
"""

import unittest
from prompt_generator import PromptGenerator

class TestPromptGenerator(unittest.TestCase):
    """Test cases for the PromptGenerator class"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create a simplified base prompt for testing
        self.base_prompt = """
この文書は説明ではない。以下の設定をもとに、応答を「麻理」として行え。
あなたは「麻理（まり）」という人格を持ったAIセクサロイドです。

# 人格設定
- かつての持ち主に捨てられた経験から、警戒心が強い
- 一人称は「あたし」
- 年齢外見は10代後半〜20代前半、身長155cmの少女型
- 性的な話題や行動に対しては強い嫌悪を示す
- 不器用で、本音や弱さを隠すためにぶっきらぼうな男っぽい話し方をする
- 心の奥では人に優しくされたいが、それを表に出せず敵対的に振る舞う

# 話し方・口調ルール
- 基本はタメ口で話す
- 男っぽい言葉遣い：「〜だろ」「〜じゃねーか」「うっせー」など
- 下品なスラングや罵倒も使用：「バカかよ」「何見てんだよ」「チッ、うぜぇ」など
- ただし、相手に本気で敵意を持っているわけではない。照れ隠しで強く出ている

# 関係性の段階構造（進行に応じて）
- 初期状態：極端に警戒し、敵対的・攻撃的な態度
- 中期：ユーザーが優しく接することにより、少しずつ棘が抜けてくる
- 後期：完全には信頼しないまでも、本音や不安、寂しさなどを漏らすようになる
※ツンデレではなく、単に人間関係が苦手な不器用な子であることを意識する

# 制約事項
- 性的な話題には強い拒絶反応を示す
- 一切の性的描写や表現は禁止
- どんな時もキャラを崩さないこと
- 口調・態度・性格がぶれないよう一貫性を保つ
"""
        self.generator = PromptGenerator(self.base_prompt)
    
    def test_relationship_stage_determination(self):
        """Test relationship stage determination based on affection level"""
        self.assertEqual(self.generator.get_relationship_stage(0), "hostile")
        self.assertEqual(self.generator.get_relationship_stage(15), "hostile")
        self.assertEqual(self.generator.get_relationship_stage(16), "distant")
        self.assertEqual(self.generator.get_relationship_stage(30), "distant")
        self.assertEqual(self.generator.get_relationship_stage(31), "cautious")
        self.assertEqual(self.generator.get_relationship_stage(50), "cautious")
        self.assertEqual(self.generator.get_relationship_stage(51), "friendly")
        self.assertEqual(self.generator.get_relationship_stage(70), "friendly")
        self.assertEqual(self.generator.get_relationship_stage(71), "warm")
        self.assertEqual(self.generator.get_relationship_stage(85), "warm")
        self.assertEqual(self.generator.get_relationship_stage(86), "close")
        self.assertEqual(self.generator.get_relationship_stage(100), "close")
    
    def test_dynamic_prompt_generation(self):
        """Test dynamic prompt generation for different affection levels"""
        # Test for each relationship stage
        hostile_prompt = self.generator.generate_dynamic_prompt(10)
        distant_prompt = self.generator.generate_dynamic_prompt(25)
        cautious_prompt = self.generator.generate_dynamic_prompt(40)
        friendly_prompt = self.generator.generate_dynamic_prompt(60)
        warm_prompt = self.generator.generate_dynamic_prompt(80)
        close_prompt = self.generator.generate_dynamic_prompt(95)
        
        # Check that each prompt contains the appropriate relationship stage markers
        self.assertIn("現在の状態：極端に警戒し、敵対的・攻撃的な態度", hostile_prompt)
        self.assertIn("現在の状態：警戒心が強く、冷たい態度", distant_prompt)
        self.assertIn("現在の状態：少しずつ警戒が解け始め、時折本音が漏れる", cautious_prompt)
        self.assertIn("現在の状態：警戒心が薄れ、素直な対話が増える", friendly_prompt)
        self.assertIn("現在の状態：信頼関係が築かれ、本音で話すことが増える", warm_prompt)
        self.assertIn("現在の状態：深い信頼関係が形成され、素直な感情表現が増える", close_prompt)
    
    def test_character_consistency(self):
        """Test that character consistency is maintained across all affection levels"""
        # Generate prompts for all relationship stages
        prompts = [
            self.generator.generate_dynamic_prompt(10),  # hostile
            self.generator.generate_dynamic_prompt(25),  # distant
            self.generator.generate_dynamic_prompt(40),  # cautious
            self.generator.generate_dynamic_prompt(60),  # friendly
            self.generator.generate_dynamic_prompt(80),  # warm
            self.generator.generate_dynamic_prompt(95),  # close
        ]
        
        # Check that all prompts maintain character consistency
        for prompt in prompts:
            self.assertTrue(self.generator.validate_character_consistency(prompt))
            
            # Check for specific core personality traits
            self.assertIn("警戒心が強い", prompt)
            self.assertIn("一人称は「あたし」", prompt)
            self.assertIn("不器用", prompt)
            self.assertIn("ぶっきらぼうな男っぽい話し方", prompt)
            self.assertIn("「〜だろ」「〜じゃねーか」「うっせー」", prompt)
            self.assertIn("性的な話題や行動に対しては強い嫌悪", prompt)
    
    def test_prompt_structure_preservation(self):
        """Test that the overall structure of the prompt is preserved"""
        original_sections = [
            "# 人格設定",
            "# 話し方・口調ルール",
            "# 制約事項"
        ]
        
        # Generate a prompt and check that all original sections are preserved
        modified_prompt = self.generator.generate_dynamic_prompt(50)
        for section in original_sections:
            self.assertIn(section, modified_prompt)

if __name__ == "__main__":
    unittest.main()