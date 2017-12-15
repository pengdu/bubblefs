//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// sql_node_visitor.h
//
// Identification: src/include/common/sql_node_visitor.h
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/include/common/sql_node_visitor.h

#ifndef BUBBLEFS_UTILS_PELOTON_SQL_NODE_VISITOR_H_
#define BUBBLEFS_UTILS_PELOTON_SQL_NODE_VISITOR_H_

namespace bubblefs {
namespace mypeloton {

namespace parser {
class SelectStatement;
class CreateStatement;
class InsertStatement;
class DeleteStatement;
class DropStatement;
class PrepareStatement;
class ExecuteStatement;
class TransactionStatement;
class UpdateStatement;
class CopyStatement;
class AnalyzeStatement;
class JoinDefinition;
struct TableRef;

class GroupByDescription;
class OrderDescription;
class LimitDescription;
}

namespace expression {
class AbstractExpression;
class ComparisonExpression;
class AggregateExpression;
class ConjunctionExpression;
class ConstantValueExpression;
class OperatorExpression;
class ParameterValueExpression;
class StarExpression;
class TupleValueExpression;
class FunctionExpression;
class OperatorUnaryMinusExpression;
class CaseExpression;
}

//===--------------------------------------------------------------------===//
// Query Node Visitor
//===--------------------------------------------------------------------===//

class SqlNodeVisitor {
 public:
  virtual ~SqlNodeVisitor(){};

  virtual void Visit(parser::SelectStatement *) {}

  // Some sub query nodes inside SelectStatement
  virtual void Visit(parser::JoinDefinition *) {}
  virtual void Visit(parser::TableRef *) {}
  virtual void Visit(parser::GroupByDescription *) {}
  virtual void Visit(parser::OrderDescription *) {}
  virtual void Visit(parser::LimitDescription *) {}

  virtual void Visit(parser::CreateStatement *) {}
  virtual void Visit(parser::InsertStatement *) {}
  virtual void Visit(parser::DeleteStatement *) {}
  virtual void Visit(parser::DropStatement *) {}
  virtual void Visit(parser::PrepareStatement *) {}
  virtual void Visit(parser::ExecuteStatement *) {}
  virtual void Visit(parser::TransactionStatement *) {}
  virtual void Visit(parser::UpdateStatement *) {}
  virtual void Visit(parser::CopyStatement *) {}
  virtual void Visit(parser::AnalyzeStatement *) {};

  virtual void Visit(expression::ComparisonExpression *expr); // expr->AcceptChildren(this);
  virtual void Visit(expression::AggregateExpression *expr);
  virtual void Visit(expression::CaseExpression *expr);
  virtual void Visit(expression::ConjunctionExpression *expr);
  virtual void Visit(expression::ConstantValueExpression *expr);
  virtual void Visit(expression::FunctionExpression *expr);
  virtual void Visit(expression::OperatorExpression *expr);
  virtual void Visit(expression::OperatorUnaryMinusExpression *expr);
  virtual void Visit(expression::ParameterValueExpression *expr);
  virtual void Visit(expression::StarExpression *expr);
  virtual void Visit(expression::TupleValueExpression *expr);

};

}  // namespace mypeloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_SQL_NODE_VISITOR_H_