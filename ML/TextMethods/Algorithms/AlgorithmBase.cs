using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Core.Distributions;

namespace ML.TextMethods.Algorithms
{
  public abstract class TextAlgorithmBase : ClassificationAlgorithmBase<string>
  {
    #region Fields

    private readonly ITextPreprocessor m_Preprocessor;
    private List<string> m_Vocabulary;
    private Dictionary<Class, double> m_PriorProbs;
    private Dictionary<Class, int>    m_ClassHist;
    private int  m_DataDim;
    private int  m_DataCount;
    private bool m_UsePriors;

    #endregion

    protected TextAlgorithmBase(ITextPreprocessor preprocessor)
    {
      if (preprocessor==null)
        throw new MLException("NaiveBayesianAlgorithmBase.ctor(preprocessor=null)");

      m_Preprocessor = preprocessor;
      m_UsePriors = true;
    }

    #region Properties

    public ITextPreprocessor Preprocessor { get { return m_Preprocessor; } }
    public List<string>      Vocabulary { get { return m_Vocabulary; } }

    public Dictionary<Class, double> PriorProbs { get { return m_PriorProbs; } }
    public Dictionary<Class, int>    ClassHist  { get { return m_ClassHist; } }
    public int DataDim             { get { return m_DataDim; } }
    public int DataCount           { get { return m_DataCount; } }

    /// <summary>
    /// If true, the algorithm takes prior class probabilities into account
    /// </summary>
    public bool UsePriors { get { return m_UsePriors; } set { m_UsePriors=value; } }

    #endregion

    public abstract double[] ExtractFeatureVector(string doc);

    public double[] ExtractFrequencies(string doc)
    {
      var dict   = Vocabulary;
      var dim    = dict.Count;
      var result = new double[dim];
      var prep   = Preprocessor;
      var tokens = prep.Preprocess(doc);

      foreach (var token in tokens)
      {
        var idx = dict.IndexOf(token);
        if (idx<0) continue;
        result[idx] += 1;
      }

      return result;
    }


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

  public abstract class NaiveBayesianAlgorithmBase : TextAlgorithmBase
  {
    private double m_Alpha;
    private bool   m_UseSmoothing;
    private Dictionary<ClassFeatureKey, double> m_Weights;

    protected NaiveBayesianAlgorithmBase(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
      m_UseSmoothing = true;
      m_Alpha = 1;
    }

    #region Properties

    public Dictionary<ClassFeatureKey, double> Weights { get { return m_Weights; } }

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
    public bool UseSmoothing { get { return m_UseSmoothing; } set { m_UseSmoothing=value; } }

    #endregion

    public override ClassScore[] PredictTokens(string obj, int cnt)
    {
      var data      = ExtractFeatureVector(obj);
      var classes   = TrainingSample.CachedClasses;
      var priors    = PriorProbs;
      var dim       = DataDim;
      var usePriors = UsePriors;
      var scores    = new List<ClassScore>();

      foreach (var cls in classes)
      {
        var score = usePriors ? Math.Log(priors[cls]) : 0.0D;
        for (int i=0; i<dim; i++)
        {
          var x = data[i];
          if (x==0) continue;
          var w = m_Weights[new ClassFeatureKey(cls, i)];
          score += x*w;
        }

        scores.Add(new ClassScore(cls, score));
      }

      return scores.OrderByDescending(s => s.Score)
                   .Take(cnt)
                   .ToArray();
    }

    public override double[] ExtractFeatureVector(string doc)
    {
      return ExtractFrequencies(doc);
    }



    protected override void TrainImpl()
    {
      m_Weights = TrainWeights();
    }

    protected abstract Dictionary<ClassFeatureKey, double> TrainWeights();
  }
}
