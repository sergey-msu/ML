using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.LogicalMethods.Model;

namespace ML.LogicalMethods.Algorithms
{
  /// <summary>
  /// Decision Tree Algorithm
  /// </summary>
  public class DecisionTreeID3Algorithm : AlgorithmBase
  {
    private DecisionTree m_Result;

    public DecisionTreeID3Algorithm(ClassifiedSample classifiedSample)
      : base(classifiedSample)
    {
    }

    public override string ID { get { return "ID3_DTREE"; } }
    public override string Name { get { return "ID3 Decision Tree"; } }

    /// <summary>
    /// Tree root node
    /// </summary>
    public DecisionTree Result { get { return m_Result; } }

    public override Class Classify(Point x)
    {
      if (m_Result==null)
        throw new MLException("Decision tree is empty");

      return m_Result.Decide(x);
    }

    /// <summary>
    /// Generate decision tree via ID3 algorithm
    /// </summary>
    public void Train(IEnumerable<Predicate<Point>> patterns, IInformIndex informativity)
    {
      if (patterns==null || !patterns.Any())
        throw new MLException("Patterns are empty or null");
      if (informativity==null)
        throw new MLException("Informativity is null");

      var root = trainID3Core(patterns, TrainingSample, informativity);

      m_Result = new DecisionTree(root);
    }

    #region .pvt

      private DecisionNode trainID3Core(IEnumerable<Predicate<Point>> patterns, ClassifiedSample sample, IInformIndex informativity)
      {
        if (!sample.Any()) throw new MLException("Empty sample");

        var cls = sample.First().Value;
        if (sample.All(kvp => kvp.Value.Equals(cls)))
          return new LeafNode(cls);

        var pattern = informativity.Max(patterns, sample);
        var negSample = new ClassifiedSample();
        var posSample = new ClassifiedSample();
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
          return new LeafNode(majorClass.Key);
        }

        var node = new InnerNode(pattern);
        var negNode = trainID3Core(patterns, negSample, informativity);
        var posNode = trainID3Core(patterns, posSample, informativity);
        node.SetNegativeNode(negNode);
        node.SetPositiveNode(posNode);

        return node;
      }

    #endregion
  }
}
