using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.LogicalMethods.Algorithms
{
  /// <summary>
  /// Decision Tree Algorithm
  /// </summary>
  public class DecisionTreeAlgorithm : AlgorithmBase
  {
    #region Inner

      /// <summary>
      /// Base class for decision tree nodes
      /// </summary>
      public abstract class Node
      {
        public abstract Class Decide(Point x);
      }

      /// <summary>
      /// Represents inner (predicate) decision tree node
      /// </summary>
      public class InnerNode : Node
      {
        private readonly Predicate<Point> m_Condition;
        private Node m_NegativeNode;
        private Node m_PositiveNode;

        public InnerNode(Predicate<Point> condition)
        {
          if (condition==null)
            throw new MLException("DecisionTree+InnerNode.ctor(condition=null)");

           m_Condition = condition;
        }

        public Node NegativeNode { get { return m_NegativeNode; } }
        public Node PositiveNode { get { return m_PositiveNode; } }

        public void SetNegativeNode(Node node)
        {
          if (node==null)
            throw new MLException("Node can not be null");

          m_NegativeNode = node;
        }

        public void SetPositiveNode(Node node)
        {
          if (node==null)
            throw new MLException("Node can not be null");

          m_PositiveNode = node;
        }

        public override Class Decide(Point x)
        {
         if (NegativeNode==null)
           throw new MLException("NegativeNode is null");
         if (PositiveNode==null)
           throw new MLException("PositiveNode is null");

          return m_Condition(x) ? PositiveNode.Decide(x) : NegativeNode.Decide(x);
        }
      }

      /// <summary>
      /// Represents leaf (class) decision tree node
      /// </summary>
      public class LeafNode : Node
      {
        private readonly Class m_Class;

        public LeafNode(Class cls)
        {
          if (cls==null)
            throw new MLException("DecisionTree+InnerNode.ctor(cls=null)");

          m_Class = cls;
        }

        public Class Class { get { return m_Class; } }

        public override Class Decide(Point x)
        {
          return m_Class;
        }
      }


    #endregion

    private Node m_Root;

    public DecisionTreeAlgorithm(ClassifiedSample classifiedSample)
      : base(classifiedSample)
    {
    }

    public override string ID { get { return "DTREE"; } }
    public override string Name { get { return "Decision Tree"; } }

    /// <summary>
    /// Tree root node
    /// </summary>
    public Node Root { get { return m_Root; } }


    public void SetRoot(Node root)
    {
      if (root==null)
        throw new MLException("Can not set null root");
      if (m_Root != null)
        throw new MLException("Root has already been set");

      m_Root = root;
    }

    public override Class Classify(Point x)
    {
      if (m_Root==null)
        throw new MLException("Decision tree is empty");

      return m_Root.Decide(x);
    }

    /// <summary>
    /// Train tree via ID3 algorithm
    /// </summary>
    public void Train_ID3(IEnumerable<Predicate<Point>> patterns, IInformIndex informativity)
    {
      if (patterns==null || !patterns.Any())
        throw new MLException("Patterns are empty or null");
      if (informativity==null)
        throw new MLException("Informativity is null");

      m_Root = trainID3Core(patterns, TrainingSample, informativity);
    }

    #region .pvt

      private Node trainID3Core(IEnumerable<Predicate<Point>> patterns, ClassifiedSample sample, IInformIndex informativity)
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
