#include "ICbcNode.hpp"

bool ICbcNode::breakTie(ICbcNode* y){
    ICbcNode* x = this;
    assert (x);
    assert (y);
      return (x->nodeNumber()>y->nodeNumber());
}
