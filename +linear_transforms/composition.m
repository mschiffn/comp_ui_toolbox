%
% composition of linear transforms (chain operation)
% author: Martin Schiffner
% date: 2016-08-13
%
classdef composition < linear_transforms.linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        N_transforms
        size_transforms
        transforms
        str_names_single
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT_composite = composition( varargin )

            % number of concatenated linear transforms
            N_transforms_temp = nargin;

            % check individual transforms for validity and size
            size = zeros( 2, N_transforms_temp );
            for index_transform = 1:N_transforms_temp

                % check class of input argument
                if ~isa( varargin{ index_transform }, 'linear_transforms.linear_transform' )
                    errorStruct.message     = sprintf( 'Input %d is not an instance of linear_transforms.linear_transform!', index_transform );
                    errorStruct.identifier	= 'LT_concatenate_diagonal:TypeMismatch';
                    error( errorStruct );
                end
                % assertion: varargin{ index_transform } is an instance of a linear transform

                % get size of selected transform
                size( :, index_transform ) = varargin{ index_transform }.size_transform;

                % create name string
                if index_transform == 1
                    str_name = sprintf( '%s', varargin{ index_transform }.str_name );
                else
                    % check correct size of linear transform for stacking
                    if size( 1, index_transform ) ~= size( 2, index_transform - 1 )
                        fprintf('incorrect size of linear transform\n');
                        return
                    end
                    str_name = sprintf( '%s_%s', str_name, varargin{ index_transform }.str_name );
                end
            end % for index_transform = 1:N_transforms_temp

            % size of diagonally stacked forward transform
            N_coefficients	= size(1, 1);
            N_lattice       = size(2, end);

            % constructor of superclass
            LT_composite@linear_transforms.linear_transform( N_coefficients, N_lattice, str_name );
            
            % internal properties
            LT_composite.N_transforms       = N_transforms_temp;
            LT_composite.transforms         = varargin;
            LT_composite.size_transforms	= size;

            LT_composite.str_names_single = cell( 1, N_transforms_temp );
            for index_transform = 1:N_transforms_temp
                LT_composite.str_names_single{ index_transform } = varargin{ index_transform }.str_name;
            end
        end

        %------------------------------------------------------------------
        % overload method: forward transform
        %------------------------------------------------------------------
        function y = forward_transform( LT_composite, x )

            % initialize internal variables
            y_temp = cell( 1, LT_composite.N_transforms );
            
            % compute chain of forward transforms
            for index_transform = LT_composite.N_transforms:-1:1

                if index_transform == LT_composite.N_transforms
                    y_temp{ index_transform } = LT_composite.transforms{index_transform}.forward_transform( x );
                else
                    y_temp{ index_transform } = LT_composite.transforms{index_transform}.forward_transform( y_temp{ index_transform + 1 } );
                end
            end

            % final result as column vector
            y = y_temp{ 1 }(:);
        end

        %------------------------------------------------------------------
        % overload method: adjoint transform
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_composite, x )

            % initialize internal variables
            y_temp = cell( 1, LT_composite.N_transforms );
            
            % compute chain of adjoint transforms
            for index_transform = 1:LT_composite.N_transforms

                if index_transform == 1
                    y_temp{ index_transform } = LT_composite.transforms{index_transform}.adjoint_transform( x );
                else
                    y_temp{ index_transform } = LT_composite.transforms{index_transform}.adjoint_transform( y_temp{ index_transform - 1 } );
                end
            end

            % final result as column vector
            y = y_temp{ end }(:);
        end

    end % methods
    
end % classdef composition < linear_transforms.linear_transform