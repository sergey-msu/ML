using System;
using ML.Core;

namespace ML.LogicalMethods.Model
{
  /// <summary>
  /// Base class for decision tree nodes
  /// </summary>
  public class DecisionTree
  {
    private readonly DecisionNode m_Root;

    public DecisionTree(DecisionNode root)
    {
      if (root==null)
        throw new MLException("DecisionTree.ctor(node=null)");

      m_Root = root;
    }

    /// <summary>
    /// Root node of the tree
    /// </summary>
    public DecisionNode Root { get { return m_Root; } }

    /// <summary>
    /// Make a decision about input object
    /// </summary>
    public virtual Class Decide(Point x)
    {
      return m_Root.Decide(x);
    }
  }
}
