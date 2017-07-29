using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.TextMethods.Algorithms
{
  public abstract class NaiveBayesianAlgorithmBase : ClassificationAlgorithmBase<string>
  {
    #region Fields

    private readonly ITextPreprocessor m_Preprocessor;
    private List<string> m_Vocabulary;
    private Dictionary<Class, double> m_PriorProbs;
    private Dictionary<Class, int> m_ClassHist;
    private int m_DataDim;
    private int m_DataCount;
    private double m_Alpha;
    private bool m_UserSmoothing;

    #endregion

    protected NaiveBayesianAlgorithmBase(ITextPreprocessor preprocessor)
    {
      if (preprocessor==null)
        throw new MLException("NaiveBayesianAlgorithmBase.ctor(preprocessor=null)");

      m_Preprocessor = preprocessor;
      m_UserSmoothing = true;
      m_Alpha = 1;
    }

    #region Properties

    public ITextPreprocessor Preprocessor { get { return m_Preprocessor; } }
    public List<string> Vocabulary { get { return m_Vocabulary; } }

    public Dictionary<Class, double> PriorProbs { get { return m_PriorProbs; } }
    public Dictionary<Class, int>    ClassHist  { get { return m_ClassHist; } }
    public int DataDim             { get { return m_DataDim; } }
    public int DataCount           { get { return m_DataCount; } }

    /// <summary>
    /// Smoothing coefficient
    /// </summary>
    public double Alpha
    {
      get { return m_Alpha; }
      set
      {
        if (value<=0)
          throw new MLException("Smoothing coefficient must be positive");

        m_Alpha=value;
      }
    }

    /// <summary>
    /// If true, uses Laplace/Lidstone smoothing
    /// </summary>
    public bool UseSmoothing { get { return m_UserSmoothing; } set { m_UserSmoothing=value; } }

    #endregion

    public override Class Predict(string obj)
    {
      var tokens = PredictTokens(obj, 1);
      if (tokens.Length <= 0) return Class.Unknown;

      return tokens[0].Class;
    }

    public abstract ClassScore[] PredictTokens(string obj, int cnt);

    public abstract double[] ExtractFeatureVector(string doc);

    protected override void DoTrain()
    {
      var corpus = TrainingSample;
      if (corpus==null || !corpus.Any())
        throw new MLException("Training sample is null or empty");

      m_Vocabulary = ExtractVocabulary(corpus);
      if (m_Vocabulary.Count<=0)
        throw new MLException("Vocabulary is empty");

      m_ClassHist  = new Dictionary<Class, int>();
      m_PriorProbs = new Dictionary<Class, double>();
      m_DataDim    = m_Vocabulary.Count;
      m_DataCount  = TrainingSample.Count;
      var classes  = TrainingSample.Classes.ToList();

      foreach (var doc in TrainingSample)
      {
        var cls = doc.Value;
        int freq;
        if (!m_ClassHist.TryGetValue(cls, out freq)) m_ClassHist[cls] = 1;
        else m_ClassHist[cls] = freq+1;
      }

      foreach (var cls in classes)
        m_PriorProbs[cls] = m_ClassHist[cls]/(double)m_DataCount;

      TrainImpl();
    }

    protected abstract void TrainImpl();

    protected virtual List<string> ExtractVocabulary(ClassifiedSample<string> corpus)
    {
      var dict = new HashSet<string>();
      foreach (var doc in corpus)
      {
        var tokens = m_Preprocessor.Preprocess(doc.Key);
        foreach (var token in tokens)
          dict.Add(token);
      }

      return dict.ToList();
    }

  }
}
