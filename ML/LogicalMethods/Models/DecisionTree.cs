using System;
using ML.Core;

namespace ML.LogicalMethods.Models
{
  /// <summary>
  /// Base class for decision tree nodes
  /// </summary>
  public class DecisionTree<TObj>
  {
    private readonly DecisionNode<TObj> m_Root;

    public DecisionTree(DecisionNode<TObj> root)
    {
      if (root==null)
        throw new MLException("DecisionTree.ctor(node=null)");

      m_Root = root;
    }

    /// <summary>
    /// Root node of the tree
    /// </summary>
    public DecisionNode<TObj> Root { get { return m_Root; } }

    /// <summary>
    /// Make a decision about input object
    /// </summary>
    public virtual Class Decide(TObj obj)
    {
      return m_Root.Decide(obj);
    }
  }
}
