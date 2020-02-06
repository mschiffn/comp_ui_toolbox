%
% abstract superclass for all spatial anti-aliasing filter options
%
% author: Martin F. Schiffner
% date: 2019-07-11
% modified: 2020-02-01
%
classdef (Abstract) anti_aliasing < scattering.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.template( size );

        end % function objects = anti_aliasing( size )

        %------------------------------------------------------------------
        % compute spatial anti-aliasing filters
        %------------------------------------------------------------------
%         function filters = compute_filter( options_anti_aliasing, axes_k_tilde, xdc_arrays, grids_FOV )
% % measurement positions
% % measurement spacing
% % measurement dimensions
%             %--------------------------------------------------------------
%             % 1.) check arguments
%             %--------------------------------------------------------------
%             % ensure class scattering.options.anti_aliasing
%             if ~isa( options_anti_aliasing, 'scattering.options.anti_aliasing' )
%                 errorStruct.message = 'options_anti_aliasing must be scattering.options.anti_aliasing!';
%                 errorStruct.identifier = 'compute_filter:NoOptionsAntiAliasing';
%                 error( errorStruct );
%             end
% 
%             % ensure equal number of dimensions and sizes
%             auxiliary.mustBeEqualSize( options_anti_aliasing, axes_k_tilde, xdc_arrays, grids_FOV );
% 
%             %--------------------------------------------------------------
%             % 2.) compute spatial anti-aliasing filters
%             %--------------------------------------------------------------
%             % specify cell array for filters
%             filters = cell( size( options_anti_aliasing ) );
% 
%             % iterate spatial anti-aliasing filter options
%             for index_options = 1:numel( options_anti_aliasing )
% 
%                 %----------------------------------------------------------
%                 % a) compute flags
%                 %----------------------------------------------------------
%                 % compute lateral components of mutual unit vectors
%                 e_r_minus_r0 = mutual_unit_vectors( math.grid( xdc_arrays( index_options ).positions_ctr ), grids_FOV( index_options ), indices_element{ index_object } );
%                 e_r_minus_r0 = repmat( abs( e_r_minus_r0( :, :, 1:(end - 1) ) ), [ N_samples_f( index_object ), 1 ] );
% 
%                 % exclude dimensions with less than two array elements
%                 indicator_dimensions = xdc_arrays( index_options ).N_elements_axis > 1;
%                 N_dimensions_lateral_relevant = sum( indicator_dimensions );
%                 e_r_minus_r0 = e_r_minus_r0( :, :, indicator_dimensions );
% 
%                 % compute flag reflecting the local angular spatial frequencies
%                 flag = real( axes_k_tilde( index_options ).members ) .* e_r_minus_r0 .* reshape( xdc_arrays( index_options ).cell_ref.edge_lengths( indicator_dimensions ), [ 1, 1, N_dimensions_lateral_relevant ] );
% 
%                 %----------------------------------------------------------
%                 % b) compute filter for current flags
%                 %----------------------------------------------------------
%                 filters{ index_object } = compute_filter_matrix( options_anti_aliasing( index_object ), flag );
% 
%             end % for index_options = 1:numel( options_anti_aliasing )
% 
%             % avoid cell array for single options_anti_aliasing
%             if isscalar( options_anti_aliasing )
%                 filters = filters{ 1 };
%             end
% 
%         end % function filters = compute_filter( options_anti_aliasing, axes_k_tilde, xdc_arrays, grids_FOV )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % compute spatial anti-aliasing filters
        %------------------------------------------------------------------
% TODO: implement in superclass and reduce method in subclasses
        filters = compute_filter( options_anti_aliasing, flags )

	end % methods (Abstract)

end % classdef (Abstract) anti_aliasing < scattering.options.template
