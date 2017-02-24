using System;
using ML.Core;

namespace ML.LogicalMethods.Model
{
  /// <summary>
  /// Base class for decision tree nodes
  /// </summary>
  public abstract class DecisionNode
  {
    public abstract Class Decide(Point x);
  }

  /// <summary>
  /// Represents inner (predicate) decision tree node
  /// </summary>
  public class InnerNode : DecisionNode
  {
    private readonly Predicate<Point> m_Condition;
    private DecisionNode m_NegativeNode;
    private DecisionNode m_PositiveNode;

    public InnerNode(Predicate<Point> condition)
    {
      if (condition == null)
        throw new MLException("DecisionTree+InnerNode.ctor(condition=null)");

      m_Condition = condition;
    }

    public DecisionNode NegativeNode { get { return m_NegativeNode; } }
    public DecisionNode PositiveNode { get { return m_PositiveNode; } }

    public void SetNegativeNode(DecisionNode node)
    {
      if (node == null)
        throw new MLException("Node can not be null");

      m_NegativeNode = node;
    }

    public void SetPositiveNode(DecisionNode node)
    {
      if (node == null)
        throw new MLException("Node can not be null");

      m_PositiveNode = node;
    }

    public override Class Decide(Point x)
    {
      if (NegativeNode == null)
        throw new MLException("NegativeNode is null");
      if (PositiveNode == null)
        throw new MLException("PositiveNode is null");

      return m_Condition(x) ? PositiveNode.Decide(x) : NegativeNode.Decide(x);
    }
  }

  /// <summary>
  /// Represents leaf (class) decision tree node
  /// </summary>
  public class LeafNode : DecisionNode
  {
    private readonly Class m_Class;

    public LeafNode(Class cls)
    {
      if (cls == null)
        throw new MLException("DecisionTree+InnerNode.ctor(cls=null)");

      m_Class = cls;
    }

    public Class Class { get { return m_Class; } }

    public override Class Decide(Point x)
    {
      return m_Class;
    }
  }

}
