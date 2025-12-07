import joblib
from django.shortcuts import render
from django.http import JsonResponse
from pathlib import Path
import pandas as pd
from django.views.decorators.csrf import csrf_exempt

# load trained model once
model_path = Path(__file__).resolve().parent.parent / "models" / "car_price_model.joblib"
model_pipeline = joblib.load(model_path)

# home page view
def home(request):
    return render(request, "index.html")  # render template

@csrf_exempt
# prediction view
def predict_car_price(request):
    if request.method == "POST":
        try:
            # get POST data
            year = int(request.POST.get("year"))
            km_driven = int(request.POST.get("km_driven"))
            fuel = request.POST.get("fuel")
            seller_type = request.POST.get("seller_type")
            transmission = request.POST.get("transmission")
            owner = request.POST.get("owner")

            # create dataframe in the same shape model expects
            input_df = pd.DataFrame({
                "year": [year],
                "km_driven": [km_driven],
                "fuel": [fuel],
                "seller_type": [seller_type],
                "transmission": [transmission],
                "owner": [owner]
            })

            # predict
            predicted_price = model_pipeline.predict(input_df)[0]

            return JsonResponse({"predicted_price": round(predicted_price)})

        except Exception as e:
            return JsonResponse({"error": str(e)})

    return JsonResponse({"error": "Invalid request"})
