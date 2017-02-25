using System;
using System.Collections.Generic;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Less-generic shortcut version of SequenceNode<TPar, TSubNode>
  ///
  ///  TPar -> TPar -> ... -> TPar
  ///
  /// </summary>
  /// <typeparam name="TPar">Input/output object type</typeparam>
  public class SequenceNode<TPar> : SequenceNode<TPar, ComputingNode<TPar, TPar>>
  {
  }

  /// <summary>
  /// Represents computing node joined from set of others -
  /// a node that sequentially tunells input through set of layers
  ///
  ///  TPar -> TPar -> ... -> TPar
  ///
  /// </summary>
  /// <typeparam name="TPar">Input/output object type</typeparam>
  /// <typeparam name="TSubNode">Type of sequential inner nodes</typeparam>
  public class SequenceNode<TPar, TSubNode> : ComputingNode<TPar, TPar>
    where TSubNode : ComputingNode<TPar, TPar>
  {
    private ParamMultiIdx m_ParIdx;
    private TSubNode[] m_SubNodes;


    public SequenceNode()
    {
      m_SubNodes = new TSubNode[0];
    }

    public override int ParamCount { get { return 0; } }
    public TSubNode[] SubNodes { get { return m_SubNodes; } }

    public void AddSubNode(TSubNode subNode)
    {
      if (subNode==null)
        throw new MLException("Node can not be null");

      var len = m_SubNodes.Length;
      var subNodes = new TSubNode[len+1];
      for (int i=0; i<len; i++)
        subNodes[i] = m_SubNodes[i];
      subNodes[len] = subNode;

      m_SubNodes = subNodes;
    }

    public override TPar Calculate(TPar input)
    {
      var result = input;
      var len = m_SubNodes.Length;
      for (int i=0; i<len; i++)
        result = m_SubNodes[i].Calculate(result);

      return result;
    }

    /// <summary>
    /// Builds node before use
    /// </summary>
    public override void DoBuild()
    {
      var len = m_SubNodes.Length;
      for (int i=0; i<len; i++)
        m_SubNodes[i].DoBuild();
    }

    /// <summary>
    /// Builds fast search index
    /// </summary>
    /// <param name="startIdx">Index start value</param>
    /// <returns>Index end value</returns>
    internal override int BuildIndex(int startIdx)
    {
      var len = m_SubNodes.Length;
      var idxs = new int[len+1];
      idxs[0] = startIdx;

      var endIdx = startIdx;
      for (int i=0; i<len; i++)
      {
        endIdx = m_SubNodes[i].BuildIndex(endIdx);
        idxs[i+1]=endIdx;
      }

      m_ParIdx = new ParamMultiIdx(idxs);

      return endIdx;
    }

    ///// <summary>
    ///// Tries to return subnode which owns parameter with specified index
    ///// WARNING: override this method carefully!
    ///// Do not use base. as base methods operate with base class index which may differ from exact class index
    ///// and so it will return wrong results. Override this method COMPLETELY or not override at all
    ///// </summary>
    ///// <param name="idx">Linear index of the parameter</param>
    ///// <param name="TNode">Subnode</param>
    ///// <param name="subidx">Internal index part within the subnode</param>
    ///// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    //public override bool TryGetSubnodeByParamIndex<TNode>(int idx, out TNode result, out int subidx)
    //{
    //  if (m_ParIdx.CheckEnd(idx))
    //  {
    //    var len = m_SubNodes.Length;
    //    for (int i=0; i<len; i++)
    //    {
    //      if (m_ParIdx.CheckIdx(idx, i+1))
    //        return m_SubNodes[i].TryGetSubnodeByParamIndex(idx, out result, out subidx);
    //    }
    //  }
    //
    //  result = null;
    //  subidx = -1;
    //  return false;
    //}

    /// <summary>
    /// Tries to return parameter value at some position
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method completely or not override
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public override bool TryGetParam(int idx, out double value)
    {
      if (m_ParIdx.CheckEnd(idx))
      {
        var len = m_SubNodes.Length;
        for (int i=0; i<len; i++)
        {
          if (m_ParIdx.CheckIdx(idx, i+1))
            return m_SubNodes[i].TryGetParam(idx, out value);
        }
      }

      value = 0;
      return false;
    }

    /// <summary>
    /// Tries to set parameter value at some position
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method completely or not override
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <param name="isDelta">Is the values are exact or just delta to existing one</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public override bool TrySetParam(int idx, double value, bool isDelta)
    {
      if (m_ParIdx.CheckEnd(idx))
      {
        var len = m_SubNodes.Length;
        for (int i=0; i<len; i++)
        {
          if (m_ParIdx.CheckIdx(idx, i+1))
            return m_SubNodes[i].TrySetParam(idx, value, isDelta);
        }
      }

      return false;
    }

    /// <summary>
    /// Tries to update parameters of the network and passes it down to all sublayers
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method COMPLETELY or not override at all
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    public override bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (pars == null || pars.Length < cursor)
        return false;

      var len = m_SubNodes.Length;
      for (int i=0; i<len; i++)
      {
        var success = m_SubNodes[i].TryUpdateParams(pars, isDelta, ref cursor);
        if (!success) return false;
      }

      return true;
    }

    protected override double DoGetParam(int idx)
    {
      return 0;
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
    }
  }
}
