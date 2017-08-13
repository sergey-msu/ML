using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.LogicalMethods.Models;

namespace ML.LogicalMethods.Algorithms
{
  /// <summary>
  /// Decision Tree Algorithm
  /// </summary>
  public class DecisionTreeID3Algorithm<TObj> : ClassificationAlgorithmBase<TObj>
  {
    private DecisionTree<TObj> m_Result;

    public DecisionTreeID3Algorithm()
    {
    }

    public override string Name { get { return "ID3_DTREE"; } }


    /// <summary>
    /// Tree root node
    /// </summary>
    public DecisionTree<TObj> Result { get { return m_Result; } }

    /// <summary>
    /// Logical patterns (predicates)
    /// </summary>
    public IEnumerable<Predicate<TObj>> Patterns { get; set; }

    /// <summary>
    /// Informativity index
    /// </summary>
    public IInformativityIndex<TObj> Informativity { get; set; }



    public override ClassScore[] PredictTokens(TObj obj, int cnt)
    {
      if (m_Result==null)
        throw new MLException("Decision tree is empty");

      var cls = m_Result.Decide(obj);
      var score = new ClassScore(cls, 1);
      return new ClassScore[] { score };
    }

    /// <summary>
    /// Generate decision tree via ID3 algorithm
    /// </summary>
    protected override void DoTrain()
    {
      base.DoTrain();

      if (Patterns==null || !Patterns.Any())
        throw new MLException("Patterns are empty or null");
      if (Informativity==null)
        throw new MLException("Informativity index is null");

      var root = trainID3Core(Patterns, TrainingSample, Informativity);

      m_Result = new DecisionTree<TObj>(root);
    }

    #region .pvt

      private DecisionNode<TObj> trainID3Core(IEnumerable<Predicate<TObj>> patterns, ClassifiedSample<TObj> sample, IInformativityIndex<TObj> informativity)
      {
        if (!sample.Any()) throw new MLException("Empty sample");

        var cls = sample.First().Value;
        if (sample.All(kvp => kvp.Value.Equals(cls)))
          return new LeafNode<TObj>(cls);

        var pattern = informativity.Max(patterns, sample);
        var negSample = new ClassifiedSample<TObj>();
        var posSample = new ClassifiedSample<TObj>();
        foreach (var pData in sample)
        {
          if (pattern(pData.Key))
            posSample.Add(pData.Key, pData.Value);
          else
            negSample.Add(pData.Key, pData.Value);
        }

        if (!negSample.Any() || !posSample.Any())
        {
          var majorClass = sample.GroupBy(pd => pd.Value)
                                 .Select(g => new KeyValuePair<Class, int>(g.Key, g.Count()))
                                 .OrderByDescending(c => c.Value)
                                 .First();
          return new LeafNode<TObj>(majorClass.Key);
        }

        var node = new InnerNode<TObj>(pattern);
        var negNode = trainID3Core(patterns, negSample, informativity);
        var posNode = trainID3Core(patterns, posSample, informativity);
        node.SetNegativeNode(negNode);
        node.SetPositiveNode(posNode);

        return node;
      }

    #endregion
  }
}
