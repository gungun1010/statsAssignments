classdef tree3 < handle
    properties
        data
        weights
        current_array_index
        split_parameters;
        max_depth;
        feature_values
        num_feats
        feats_max
        feats_min
    end
    methods
        function obj = tree3(feature_values,max_dep,weights)
            data_length = length(feature_values(:,1));
            obj.data(1,1:2) = [0,0];
            obj.data(1,3)=1;
            obj.data(1,4)=0;
            obj.data(1,5) = data_length;
            obj.data(1,6) =0;
            obj.data(1,7:(6+data_length)) = 1:data_length;
            
            obj.current_array_index =1;
            
            obj.max_depth = max_dep;
            
            obj.feature_values = feature_values;
            
            obj.weights = weights;
            
            temp=size(feature_values,2)-1;
            obj.num_feats = temp;
            
            for j=1:temp
                obj.feats_max(j) = max(feature_values(j+1,:));
                obj.feats_min(j) = min(feature_values(j+1,:));
            end
        end
        
        function [] = add_nodes(obj, base_node,data1, data2, opt_param)
            obj.data(base_node,1)=obj.current_array_index+1;
            obj.data(base_node,2)=obj.current_array_index+2;
            
            obj.split_parameters(base_node,:) = opt_param;
            
            base_depth = obj.data(base_node, 3);
            
            obj.data(obj.current_array_index+1, 1:2) = [0,0];
            obj.data(obj.current_array_index+1, 3) = base_depth+1;
            obj.data(obj.current_array_index+1,4) = base_node;
            obj.data(obj.current_array_index+1, 5) = length(data1);
            obj.data(obj.current_array_index+1, 6) = 0;
            obj.data(obj.current_array_index+1, 7: (6+length(data1))) = data1;
            
            obj.data(obj.current_array_index+2, 1:2) = [0,0];
            obj.data(obj.current_array_index+2, 3) = base_depth+1;
            obj.data(obj.current_array_index+2, 4) = base_node;
            obj.data(obj.current_array_index+2, 5) = length(data2);
            obj.data(obj.current_array_index+2, 6) = 0;
            obj.data(obj.current_array_index+2, 7: (6+length(data2))) = data2;
            
            obj.current_array_index = obj.current_array_index +2;
            
            return
        end
        
        function out = objective(obj,data,param)
            
            [data1,data2] = obj.split(data, param);
            
            len = length(data);
            len1 = length(data1);
            len2 = length(data2);
            
            if(len1 ==0 || len2 ==0)
                out = Inf;
                return
            end
            
            out=0;
            
            for i=1:len1
                if(param(2) ~= obj.feature_values(data1(i),1))
                    out = out + obj.weights(data1(i));
                end
            end
            
            for i=1:len2
                if(param(3) ~= obj.feature_values(data2(i),1))
                    out = out + obj.weights(data2(i));
                end
            end
            
            return
        end
        
        function out = weak_learner(obj,data_point,param)
            
            if(data_point(param(1))<param(4))
                out=0;
            else
                out=1;
            end
            
            return
            
        end
        
        function [data1,data2] = split(obj, data_in, param)
            data1_counter=1;
            data2_counter=1;
            data1=[];
            data2=[];
            
            len_feat_val = obj.num_feats;
            
            for i=1:length(data_in)
                if(weak_learner(obj, obj.feature_values(data_in(i),2:(1+len_feat_val)), param)==0)
                    data1(data1_counter) = data_in(i);
                    data1_counter = data1_counter +1;
                else
                    data2(data2_counter) = data_in(i);
                    data2_counter = data2_counter +1;
                end
            end
            
            return
        end
        
        function opt_param = optimize_parameter(obj,data_points)
            
            opt_param = obj.random_parameter();
            opt_objective = obj.objective(data_points,opt_param);
            
            for i=1:200
                r_param = obj.random_parameter();
                temp = obj.objective(data_points, r_param);
                
                if(temp < opt_objective)
                    opt_objective = temp;
                    opt_param = r_param;
                end
            end
            
            return
        end
        
        function random_param = random_parameter(obj)
            
            feature = unidrnd(obj.num_feats);
            
            label1 = unidrnd(2);
            if(label1==1)
                label1=-1;
            else
                label1=1;
            end
            
            label2 = unidrnd(2);
            if(label2==1)
                label2=-1;
            else
                label2=1;
            end
            
            tau = (rand*(obj.feats_max(feature) - obj.feats_min(feature)))+obj.feats_min(feature);
            
            random_param = [feature,label1, label2,tau];
            
            return
        end
        
        function labels = extract_labels(obj, data_points)
            labels=[];
            for i=1:length(data_points)
                labels(i) = obj.feature_values(data_points(i),1);
            end
            return
        end
        
        function [] = train(obj)
            flag = 1;
            while(flag~=0)
                flag = 0;
                for i=1:length(obj.data(:,1))
                    if(obj.data(i,6) ~= -1)
                        if( obj.data(i,3) >= obj.max_depth)
                            obj.data(i,6)=-1;
                        else
                            data_len = obj.data(i,5);
                            dat = obj.data(i,7:(6+data_len));
                            opt_param = obj.optimize_parameter(dat);
                            [data1,data2] = split(obj, obj.data(i, 7:6+data_len), opt_param);
                            if(length(data1) ~=0 && length(data2 ~=0))
                                obj.add_nodes(i,data1, data2,opt_param);
                            end
                            obj.data(i,6)=-1;
                            flag = flag+1;
                        end
                    end
                end
                
            end
            
            return
        end
        
        function out = test(obj, data_point)
            
            data_len = length(data_point(:,1));
            out = zeros(1,data_len);
            for data_counter=1:data_len
                
                current_node=1;
                
                while(obj.data(current_node, 1) ~= 0)
                    split_param = obj.split_parameters(current_node,:);
                    if(obj.weak_learner(data_point(data_counter,:), split_param)==0)
                        current_node = obj.data(current_node, 1);
                    else
                        current_node = obj.data(current_node,2) ;
                    end
                end
                
                parent_index = obj.data(current_node,4);
                
                if(obj.data(parent_index,1)==current_node)
                    out(data_counter) = obj.split_parameters(parent_index, 2);
                else
                    out(data_counter) = obj.split_parameters(parent_index,3);
                end
                
            end
            return
        end
        
    end
end