%
% abstract superclass for all discrete convolutions
%
% author: Martin F. Schiffner
% date: 2019-12-08
% modified: 2020-10-26
%
classdef (Abstract) convolution < linear_transforms.linear_transform_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % independent properties
        kernel ( :, 1 )                                     % convolution kernel
        cut_off ( 1, 1 ) logical { mustBeNonempty } = true	% cut off results to ensure square matrix

        % dependent properties
        M_kernel ( 1, 1 ) double { mustBePositive, mustBeNonempty } = 1	% symmetric length of kernel

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = convolution( kernels, N_points, cut_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure valid number of input arguments
            narginchk( 2, 3 );

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
            if nargin < 3 || isempty( cut_off )
                cut_off = true;
            end

            % property validation function ensures logical for cut_off

            % ensure equal number of dimensions and sizes
            [ kernels, N_points, cut_off ] = auxiliary.ensureEqualSize( kernels, N_points, cut_off );

            %--------------------------------------------------------------
            % 2.) create discrete convolutions
            %--------------------------------------------------------------
            % symmetric lengths of kernels
            N_kernel = cellfun( @numel, kernels );
            M_kernel = ( N_kernel - 1 ) / 2;

            % ensure integers if cut_off are true
            mustBeInteger( M_kernel( cut_off ) );

            % numbers of coefficients
            N_coefficients = N_points + N_kernel - 1;
            N_coefficients( cut_off ) = N_points;

            % constructor of superclass
            objects@linear_transforms.linear_transform_matrix( N_coefficients, N_points );

            % truncate kernel
            indicator_cut_kernel = cut_off & ( ( M_kernel + 1 ) > N_points );

            % iterate discrete convolutions
            for index_object = 1:numel( objects )

                % set independent properties
                if indicator_cut_kernel( index_object )
                    % truncate kernel
                    M_cut = M_kernel( index_object ) - N_points( index_object ) + 1;
                    kernels{ index_object } = kernels{ index_object }( ( M_cut + 1 ):(end - M_cut) );
                    M_kernel( index_object ) = N_points( index_object ) - 1;

                    objects( index_object ).kernel = kernels{ index_object };

                else
                    % original kernel
                    objects( index_object ).kernel = kernels{ index_object };
                end
                objects( index_object ).cut_off = cut_off( index_object );

                % set dependent properties
                objects( index_object ).M_kernel = M_kernel( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = convolution( kernels, N_points, cut_off )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % display coefficients (single matrix)
        %------------------------------------------------------------------
        function display_coefficients_matrix( LT, x )
        end % function display_coefficients_matrix( LT, x )

        %------------------------------------------------------------------
        % relative RMSEs of best s-sparse approximations (single matrix)
        %------------------------------------------------------------------
        function [ rel_RMSEs, axes_s ] = rel_RMSE_matrix( LT, y )
        end

	end % methods (Access = protected, Hidden)

end % classdef (Abstract) convolution < linear_transforms.linear_transform_matrix
