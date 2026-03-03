
# Test script to verify Stage2Verifier with AASIST_Backend works correctly

import torch
import sys
sys.path.insert(0, r'c:\Users\ritam\Desktop\voice_detector\backend')

from aasist_backend import AASIST_Backend

def test_aasist_backend():
    """Test AASIST_Backend forward pass"""
    print("Testing AASIST_Backend...")
    
    # Create dummy input (batch=2, time=100, features=768)
    features = torch.randn(2, 100, 768)
    
    # Initialize backend
    backend = AASIST_Backend(input_dim=768)
    backend.eval()
    
    # Forward pass
    with torch.no_grad():
        output = backend(features)
    
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"
    print("[PASS] AASIST_Backend test passed!")
    
    return True

def test_stage2_verifier():
    """Test Stage2Verifier with AASIST_Backend"""
    print("\nTesting Stage2Verifier...")
    
    try:
        from final_inference_logic import Stage2Verifier
        
        # Create model
        model = Stage2Verifier()
        model.eval()
        
        # Create dummy audio input (batch=1, samples=64000 which is 4 seconds at 16kHz)
        audio = torch.randn(1, 64000)
        
        # Forward pass
        with torch.no_grad():
            output = model(audio)
        
        print(f"Audio shape: {audio.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output value: {output.item()}")
        
        # Apply sigmoid to get probability
        prob = torch.sigmoid(output).item()
        print(f"Probability (after sigmoid): {prob:.4f}")
        
        assert output.shape == (1,) or output.shape == torch.Size([]), \
            f"Expected scalar output, got {output.shape}"
        print("[PASS] Stage2Verifier test passed!")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Stage2Verifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("AASIST Backend Integration Test")
    print("="*60)
    
    # Test 1: AASIST_Backend standalone
    try:
        test_aasist_backend()
    except Exception as e:
        print(f"[FAIL] AASIST_Backend test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Stage2Verifier with AASIST_Backend
    test_stage2_verifier()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
