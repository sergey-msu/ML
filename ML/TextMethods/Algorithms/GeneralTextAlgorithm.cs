using System;
using System.Collections.Generic;
using ML.Core;
using ML.Contracts;

namespace ML.TextMethods.Algorithms
{
  public class GeneralTextAlgorithm : TextAlgorithmBase
  {
    private readonly IClassificationAlgorithm<double[]> m_SubAlgorithm;

    public GeneralTextAlgorithm(ITextPreprocessor preprocessor, IClassificationAlgorithm<double[]> alg)
      : base(preprocessor)
    {
      if (alg==null) throw new MLException("GeneralNaiveBayesianAlgorithm.ctor(alg=null)");

      m_SubAlgorithm = alg;
    }

    #region Properties

    public override string Name { get { return "GENNB"; } }

    public IClassificationAlgorithm<double[]> SubAlgorithm { get { return m_SubAlgorithm; } }

    #endregion

    public override ClassScore[] PredictTokens(string obj, int cnt)
    {
      var data = ExtractFeatureVector(obj);
      return m_SubAlgorithm.PredictTokens(data, cnt);
    }

    public override double[] ExtractFeatureVector(string doc)
    {
      return ExtractFrequencies(doc);
    }


    protected override void TrainImpl()
    {
      var featureSample = new ClassifiedSample<double[]>();
      foreach (var pData in TrainingSample)
      {
        var doc  = pData.Key;
        var data = ExtractFeatureVector(doc);
        var cls  = pData.Value;
        featureSample[data] = cls;
      }

      m_SubAlgorithm.Train(featureSample);
    }
  }
}
