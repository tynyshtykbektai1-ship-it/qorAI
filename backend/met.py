import os
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_analyzer(analyzer, test_dir):
    """
    test_dir должна содержать подпапки 'healthy' и 'mastitis'
    """
    y_true = []
    y_pred = []
    
    # Сопоставляем имена папок с метками (0 - здорова, 1 - мастит)
    categories = {'healthy': 0, 'mastitis': 1}
    
    print("Начинаю валидацию...")
    
    for cat_name, label in categories.items():
        cat_path = os.path.join(test_dir, cat_name)
        if not os.path.exists(cat_path): continue
        
        for img_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                result = analyzer.predict(img)
                
                # Если модель сказала, что это не сосок, пропускаем или 
                # помечаем как ошибку (зависит от вашей задачи)
                if not result['is_valid']:
                    continue 
                
                pred_label = 1 if result['status'] == "Mastitis Detected" else 0
                y_true.append(label)
                y_pred.append(pred_label)
            except Exception as e:
                print(f"Ошибка в файле {img_name}: {e}")

    # Печать метрик
    print("\n=== Результаты классификации мастита ===")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Mastitis']))