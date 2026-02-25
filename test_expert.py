from src.processing_engine import ProcessingEngine

expert_folder = '/Users/surya_rayala/Desktop/Projects/Army-Devcom/GIFT/Archive/GIFT-EAE-ECR/GIFT-Multimodal-EAE-main/expert_compare_test/expert'
session_folder = '/Users/surya_rayala/Desktop/Projects/Army-Devcom/GIFT/Archive/GIFT-EAE-ECR/GIFT-Multimodal-EAE-main/expert_compare_test/trainee'
metric_name = "TOTAL_FLOOR_COVERAGE_TIME"
vmeta_path = '/Users/surya_rayala/Desktop/Projects/Army-Devcom/GIFT/GIFT-Multimodal-EAE-main/input/test.vmeta.xml'
engine = ProcessingEngine()
print(engine.compare_expert(metric_name, session_folder, expert_folder, vmeta_path))