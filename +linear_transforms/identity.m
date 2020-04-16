%
% compute identity for various options
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-01-30
%
classdef identity < linear_transforms.linear_transform_matrix

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = identity( N_points )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures nonempty positive integers for N_points

            %--------------------------------------------------------------
            % 2.) create identity operators
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.linear_transform_matrix( N_points, N_points );

        end % function objects = identity( N_points )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward transform (single matrix)
        %------------------------------------------------------------------
        function x = forward_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.identity (scalar)
            if ~( isa( LT, 'linear_transforms.identity' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.identity!';
                errorStruct.identifier = 'forward_transform_matrix:NoSingleIdentity';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward identity (single matrix)
            %--------------------------------------------------------------
            % output x equals argument x

        end % function x = forward_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single matrix)
        %------------------------------------------------------------------
        function y = adjoint_transform_matrix( LT, x )

            % adjoint transform equals forward transform
            y = forward_transform_matrix( LT, x );

        end % function y = adjoint_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % inverse transform (single matrix)
        %------------------------------------------------------------------
        function y = inverse_transform_matrix( LT, x )

            % inverse transform equals forward transform
            y = forward_transform_matrix( LT, x );

        end % function y = inverse_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % display coefficients (single matrix)
        %------------------------------------------------------------------
        function display_coefficients_matrix( LT, x )
        end % function display_coefficients_matrix( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef identity < linear_transforms.linear_transform_matrix
