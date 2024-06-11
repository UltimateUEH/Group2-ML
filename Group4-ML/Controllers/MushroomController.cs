using System.Diagnostics;
using System.IO;
using Group4_ML.Models;
using Microsoft.AspNetCore.Mvc;
using Python.Runtime;

namespace Group2_ML.Controllers
{
    public class MushroomController : Controller
    {
        public IActionResult Input()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Predict(MushroomInputModel inputModel)
        {
            if (ModelState.IsValid)
            {
                var predictionResult = PredictWithPythonModel(inputModel);
                return View("Result", new PredictionResult { Result = predictionResult });
            }

            return View("Input", inputModel);
        }

        private string PredictWithPythonModel(MushroomInputModel inputModel)
        {
            string result;
            string pythonScriptPath = Path.Combine(Directory.GetCurrentDirectory(), "PythonScripts", "predict.py");

            // Setup Python runtime
            PythonEngine.PythonHome = @"C:\Users\ADMIN\AppData\Local\Programs\Python\Python37"; // Adjust this path to your Python installation
            PythonEngine.PythonPath = Environment.GetEnvironmentVariable("PYTHONPATH") +
                                      Path.PathSeparator +
                                      Path.Combine(PythonEngine.PythonHome, "Lib") +
                                      Path.PathSeparator +
                                      Path.Combine(PythonEngine.PythonHome, "DLLs");
            PythonEngine.Initialize();

            using (Py.GIL())
            {
                dynamic py = Py.Import("predict");
                result = py.predict(
                    inputModel.CapShape,
                    inputModel.CapSurface,
                    inputModel.CapColor,
                    inputModel.Bruises,
                    inputModel.Odor,
                    inputModel.GillAttachment,
                    inputModel.GillSpacing,
                    inputModel.GillSize,
                    inputModel.GillColor,
                    inputModel.StalkShape,
                    inputModel.StalkRoot,
                    inputModel.StalkSurfaceAboveRing,
                    inputModel.StalkSurfaceBelowRing,
                    inputModel.StalkColorAboveRing,
                    inputModel.StalkColorBelowRing,
                    inputModel.VeilType,
                    inputModel.VeilColor,
                    inputModel.RingNumber,
                    inputModel.RingType,
                    inputModel.SporePrintColor,
                    inputModel.Population,
                    inputModel.Habitat
                );
            }

            // Shutdown Python engine if necessary
            PythonEngine.Shutdown();

            return result;
        }
    }
}
