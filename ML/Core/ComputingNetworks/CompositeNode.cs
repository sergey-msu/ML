using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Less-generic shortcut version of AggreagteNode<TIn, TOut, TSubNode>
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  public class CompositeNode<TIn, TOut> : CompositeNode<TIn, TOut, ComputingNode<TIn, TOut>>
  {
  }

  /// <summary>
  /// Represents composite computing node -
  /// a node that composed from another nodes, calculates the results (may be in parallel) and composes its outputs in one vector
  ///
  ///       TIn -> TOut
  ///     /
  /// TIn - TIn -> TOut -> TOut[]
  ///     \
  ///       TIn -> TOut
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  /// <typeparam name="TOut">Type of subnodes</typeparam>
  public class CompositeNode<TIn, TOut, TSubNode> : ComputingNode<TIn, TOut[]>
    where TSubNode : ComputingNode<TIn, TOut>
  {
    public const int DFT_MAX_DEGREES_OF_PARALLELISM = 8;

    private ParamMultiIdx m_ParIdx;
    private TSubNode[] m_SubNodes;

    public CompositeNode()
    {
      m_SubNodes = new TSubNode[0];
    }

    public override int ParamCount { get { return 0; } }
    public TSubNode[] SubNodes { get { return m_SubNodes; } }
    public bool IsParallel { get; set; }
    public int MaxDegreeOfParallelism { get; set; }

    /// <summary>
    /// Adds Subnode to inner node aggregation list
    /// </summary>
    public void AddSubNode(TSubNode subNode)
    {
      if (subNode==null)
        throw new MLException("Subnode can not be null");

      var len = m_SubNodes.Length;
      var subNodes = new TSubNode[len+1];
      for (int i=0; i<len; i++)
        subNodes[i] = m_SubNodes[i];
      subNodes[len] = subNode;

      m_SubNodes = subNodes;
    }

    /// <summary>
    /// Calculates final node result as some aggregation('merge') of subnodes results.
    /// May run in parallel.
    /// </summary>
    public override TOut[] Calculate(TIn input)
    {
      var len = m_SubNodes.Length;
      var results = new TOut[len];

      if (IsParallel)
      {
        if (MaxDegreeOfParallelism <= 0) MaxDegreeOfParallelism = DFT_MAX_DEGREES_OF_PARALLELISM;
        var opts = new ParallelOptions { MaxDegreeOfParallelism = MaxDegreeOfParallelism };

        Parallel.For(0, len, opts, i =>
        {
          results[i] = m_SubNodes[i].Calculate(input);
        });
      }
      else
      {
        for (int i = 0; i < len; i++)
          results[i] = m_SubNodes[i].Calculate(input);
      }

      return results;
    }

    /// <summary>
    /// Builds node before use
    /// </summary>
    public override void DoBuild()
    {
      if (m_SubNodes.Length <= 0)
        throw new MLException("Subnodes has not been set");

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
      if (m_SubNodes.Length <= 0)
        throw new MLException("Subnodes has not been set");

      var len = m_SubNodes.Length;
      var idxs = new int[len+1];
      idxs[0] = startIdx;

      var endIdx = startIdx;

      for (int i=0; i<len; i++)
      {
        endIdx = m_SubNodes[i].BuildIndex(endIdx);
        idxs[i+1] = endIdx;
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

      var success = false;
      var len = m_SubNodes.Length;
      for (int i=0; i<len; i++)
      {
        success = m_SubNodes[i].TryUpdateParams(pars, isDelta, ref cursor);
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
