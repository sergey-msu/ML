using System;
using ML.Core;

namespace ML.LogicalMethods.Models
{
  /// <summary>
  /// Base class for decision tree nodes
  /// </summary>
  public abstract class DecisionNode<TObj>
  {
    public abstract Class Decide(TObj x);
  }

  /// <summary>
  /// Represents inner (predicate) decision tree node
  /// </summary>
  public class InnerNode<TObj> : DecisionNode<TObj>
  {
    private readonly Predicate<TObj> m_Condition;
    private DecisionNode<TObj> m_NegativeNode;
    private DecisionNode<TObj> m_PositiveNode;

    public InnerNode(Predicate<TObj> condition)
    {
      if (condition == null)
        throw new MLException("DecisionTree+InnerNode.ctor(condition=null)");

      m_Condition = condition;
    }

    public DecisionNode<TObj> NegativeNode { get { return m_NegativeNode; } }
    public DecisionNode<TObj> PositiveNode { get { return m_PositiveNode; } }

    public void SetNegativeNode(DecisionNode<TObj> node)
    {
      if (node == null)
        throw new MLException("Node can not be null");

      m_NegativeNode = node;
    }

    public void SetPositiveNode(DecisionNode<TObj> node)
    {
      if (node == null)
        throw new MLException("Node can not be null");

      m_PositiveNode = node;
    }

    public override Class Decide(TObj x)
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
  public class LeafNode<TObj> : DecisionNode<TObj>
  {
    private readonly Class m_Class;

    public LeafNode(Class cls)
    {
      if (cls.IsUnknown)
        throw new MLException("DecisionTree+InnerNode.ctor(cls.IsUnknown)");

      m_Class = cls;
    }

    public Class Class { get { return m_Class; } }

    public override Class Decide(TObj x)
    {
      return m_Class;
    }
  }

}
