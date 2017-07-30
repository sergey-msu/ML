using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Core.Distributions;

namespace ML.TextMethods.Algorithms
{
  public class GeneralNaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
  {
    private Dictionary<ClassFeatureKey, double> m_Frequencies;

    public GeneralNaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
    }

    #region Properties

    public override string Name   { get { return "TWCNB"; } }

    #endregion

    public override ClassScore[] PredictTokens(string obj, int cnt)
    {
      return null;
    }

    public override double[] ExtractFeatureVector(string doc)
    {
      return null;
    }


    protected override void TrainImpl()
    {
    }


  }
}
