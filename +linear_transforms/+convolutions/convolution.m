%
% abstract superclass for all discrete convolutions
%
% author: Martin F. Schiffner
% date: 2019-12-08
% modified: 2020-04-03
%
classdef (Abstract) convolution < linear_transforms.linear_transform_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % independent properties
        kernel ( :, 1 )
        q_lb ( 1, 1 ) double { mustBeNonnegative, mustBeNonempty } = 0      % lower bound on the indices of admissible frequencies
        cut_off ( 1, 1 ) logical { mustBeNonempty } = true                  % cut off results to ensure square matrix

        % dependent properties
        M_kernel ( 1, 1 ) double { mustBePositive, mustBeNonempty } = 1     % symmetric length of kernel

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = convolution( kernels, N_points, q_lb, cut_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure correct number of input arguments
            narginchk( 2, 4 );

            % ensure cell array for kernels
            if ~iscell( kernels )
                kernels = { kernels };
            end

            % ensure column vectors for kernels
            indicator = cellfun( @( x ) ~iscolumn( x ), kernels );
            if any( indicator( : ) )
                errorStruct.message = 'kernels must contain column vectors!';
                errorStruct.identifier = 'convolution:NoColumnVectors';
                error( errorStruct );
            end

            % superclass ensures nonempty positive integers for N_points

            % ensure nonempty cut_off
            if nargin < 4 || isempty( cut_off )
                cut_off = true;
            end

            % ensure nonempty cut_off
            if nargin < 4 || isempty( cut_off )
                cut_off = true;
            end

            % property validation function ensures logical for cut_off

            % multiple kernels / single N_points
            if ~isscalar( kernels ) && isscalar( N_points )
                N_points = repmat( N_points, size( kernels ) );
            end

            % multiple kernels / single cut_off
            if ~isscalar( kernels ) && isscalar( cut_off )
                cut_off = repmat( cut_off, size( kernels ) );
            end

            % multiple N_points / single kernels
            if ~isscalar( N_points ) && isscalar( kernels )
                kernels = repmat( kernels, size( N_points ) );
            end

            % multiple N_points / single cut_off
            if ~isscalar( N_points ) && isscalar( cut_off )
                cut_off = repmat( cut_off, size( N_points ) );
            end

            % multiple cut_off / single kernels
            if ~isscalar( cut_off ) && isscalar( kernels )
                kernels = repmat( kernels, size( cut_off ) );
            end

            % multiple cut_off / single N_points
            if ~isscalar( cut_off ) && isscalar( N_points )
                N_points = repmat( N_points, size( cut_off ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( kernels, N_points, cut_off );

            %--------------------------------------------------------------
            % 2.) create discrete convolutions
            %--------------------------------------------------------------
            % symmetric lengths of kernels
            M_kernel = ( cellfun( @numel, kernels ) - 1 ) / 2;

            % ensure integers if cut_off are true
            mustBeInteger( cut_off .* M_kernel );

            % number of coefficients
            N_coefficients = ~cut_off .* ( cellfun( @numel, kernels ) - 1 ) + N_points;

            % constructor of superclass
            objects@linear_transforms.linear_transform_matrix( N_coefficients, N_points );

            % iterate discrete convolutions
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).kernel = kernels{ index_object };
                objects( index_object ).cut_off = cut_off( index_object );

                % set dependent properties
                objects( index_object ).M_kernel = M_kernel( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = convolution( kernels, N_points, cut_off )

    end % methods

end % classdef (Abstract) convolution < linear_transforms.linear_transform_matrix
