#!/usr/bin/env python3
"""
Kosmos Trainer 패치 - tokenize 문제 해결
"""

import types
import torch

def patch_trainer(trainer):
    """기존 트레이너의 train_step 메서드를 패치"""
    
    def patched_train_step(self, batch):
        """패치된 train_step - tokenize 문제 해결"""
        self.model.train()
        
        # 입력 데이터 준비
        pixel_values = batch["vision_x"].to(self.device)  # [B, T, 3, 224, 224]
        
        # 명령어 토크나이징 - 안전한 처리
        task_desc = batch["task_description"]
        
        # task_description 타입에 따른 안전한 처리
        if isinstance(task_desc, str):
            instructions = [task_desc]
        elif isinstance(task_desc, (list, tuple)):
            instructions = list(task_desc)
        else:
            # 기타 경우 문자열로 변환
            instructions = [str(task_desc)]
        
        # 빈 문자열이나 None 체크
        instructions = [instr for instr in instructions if instr and instr.strip()]
        if not instructions:
            instructions = ["Navigate to track the target cup"]  # 기본 명령어
        
        print(f"🔍 Instructions: {instructions}")  # 디버깅
        
        # 토크나이징
        try:
            tokenized = self.tokenize_instructions(instructions)
        except Exception as e:
            print(f"❌ Tokenize 오류: {e}")
            print(f"   Instructions type: {type(instructions)}")
            print(f"   Instructions content: {instructions}")
            raise e
        
        # 순전파
        predictions = self.model(
            pixel_values=pixel_values,
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask")
        )
        
        # 타겟 준비 (Mobile VLA 원본 데이터 사용)
        targets = {
            "mobile_actions": batch["mobile_actions"].to(self.device),  # [B, T, 3] - Mobile VLA 액션
            "mobile_events": batch["mobile_events"].to(self.device)     # [B, T] - Mobile VLA 이벤트
        }
        
        # 손실 계산
        losses = self.compute_loss(predictions, targets)
        
        return losses
    
    # 패치 적용
    trainer.train_step = types.MethodType(patched_train_step, trainer)
    print("✅ Kosmos Trainer 패치 완료!")
    
    return trainer


if __name__ == "__main__":
    print("🔧 Trainer 패치 유틸리티")
