using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using ML.Core;

namespace ML.ConsoleTest
{
  public class DataWrapper
  {
    #region Inner

      public class DataError
      {
        public int    LineNum { get; set; }
        public string Line { get; set; }
      }

    #endregion

    private const string CLASS_HEADER    = "_class";
    private const string CLASS_VALUE     = "_value";
    private const string TRAINING_HEADER = "_training";

    public DataWrapper(string resourceName)
    {
      readData(resourceName);
    }

    public readonly Dictionary<string, Class>   Classes = new Dictionary<string, Class>();
    public readonly Dictionary<string, Feature> Features = new Dictionary<string, Feature>();
    public readonly ClassifiedSample<double[]>  Data = new ClassifiedSample<double[]>();
    public readonly ClassifiedSample<double[]>  TrainingSample = new ClassifiedSample<double[]>();
    public readonly List<DataError> Errors = new List<DataError>();

    public int Dimension { get; private set; }

    private void readData(string file)
    {
      var assembly = Assembly.GetExecutingAssembly();
      var resourceName = string.Format("{0}.Data.{1}", assembly.GetName().Name, file);

      using (var stream = assembly.GetManifestResourceStream(resourceName))
      using (var reader = new StreamReader(stream))
      {
        int trainingIndx;
        int classesIndx;
        int clsValIdx;
        int[] featureIndxs;

        try
        {
          readHeaders(reader, out featureIndxs, out trainingIndx, out classesIndx, out clsValIdx);
          readBody(reader, featureIndxs, trainingIndx, classesIndx, clsValIdx);
          Dimension = featureIndxs.Length;
        }
        catch (Exception ex)
        {
          Console.WriteLine("CRITICAL ERROR: {0}", ex.Message);
        }
      }

    }

    private void readHeaders(StreamReader reader, out int[] featureIndxs, out int trainingIndx, out int classesIndx, out int clsValIdx)
    {
      string line;
      do { line = reader.ReadLine(); }
      while (line.StartsWith("//"));

      var headers = line.Split(',');

      var trIdx = Array.IndexOf(headers, TRAINING_HEADER);
      var clIdx = Array.IndexOf(headers, CLASS_HEADER);
      var vIdx = Array.IndexOf(headers, CLASS_VALUE);

      if (clIdx < 0)
        throw new InvalidOperationException("Invalid data header: last class column not found");

      var ftIdxs = Enumerable.Range(0, headers.Length).Where(i => i != trIdx &&
                                                                  i != clIdx &&
                                                                  i != vIdx);
      if (ftIdxs.Count() < 1)
        throw new InvalidOperationException("Invalid data header: features not found");
      foreach (var idx in ftIdxs)
        Features.Add(headers[idx], new Feature(headers[idx]));

      trainingIndx = trIdx;
      classesIndx = clIdx;
      clsValIdx = vIdx;
      featureIndxs = ftIdxs.ToArray();
    }

    private void readBody(StreamReader reader, int[] featureIndxs, int trainingIndx, int classesIndx, int clsValIdx)
    {
      var dim = featureIndxs.Length;
      var lineNum = 0;

      while (true)
      {
        lineNum++;
        var line = reader.ReadLine();
        if (line != null && line.StartsWith("//")) continue;
        if (string.IsNullOrWhiteSpace(line)) break;
        var data = line.Split(',');

        var success = true;
        var point = new double[dim];
        for (var i=0; i<dim; i++)
        {
          double result;
          var ftIdx = featureIndxs[i];
          if(!double.TryParse(data[ftIdx], out result))
          {
            success = false;
            break;
          }
          point[i] = result;
        }

        if (!success)
        {
          Errors.Add(new DataError { LineNum = lineNum, Line = line });
          continue;
        }
        Class cls;
        var clsName = data[classesIndx];
        if (!Classes.TryGetValue(clsName, out cls))
        {
          double val;
          var value = (clsValIdx<0 || !double.TryParse(data[clsValIdx], out val)) ? (double?)null : val;
          cls = new Class(clsName, value);
          Classes[clsName] = cls;
        }

        Data.Add(point, cls);

        if (trainingIndx >= 0)
        {
          var isTraining = int.Parse(data[trainingIndx]) != 0;
          if (isTraining)
            TrainingSample.Add(point, cls);
        }
      }
    }

  }
}
