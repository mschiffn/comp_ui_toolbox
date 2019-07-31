function h_transfer_aa = anti_aliasing_filter( xdc_arrays, homogeneous_fluids, h_transfer, options_anti_aliasing, varargin )
%
% apply anti-aliasing filter to
% the spatial transfer function for the d-dimensional Euclidean space
%
% author: Martin F. Schiffner
% date: 2019-07-10
% modified: 2019-07-29
%

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure class transducers.array_planar_regular_orthogonal
	if ~isa( xdc_arrays, 'transducers.array_planar_regular_orthogonal' )
        errorStruct.message = 'xdc_arrays must be transducers.array_planar_regular_orthogonal!';
        errorStruct.identifier = 'anti_aliasing_filter:NoSingleOperatorBorn';
        error( errorStruct );
    end

    % ensure class discretizations.field
    if ~isa( h_transfer, 'discretizations.field' )
        errorStruct.message = 'h_transfer must be discretizations.field!';
        errorStruct.identifier = 'anti_aliasing_filter:NoSingleOperatorBorn';
        error( errorStruct );
    end

	% ensure nonempty indices_element
	if nargin >= 5 && ~isempty( varargin{ 1 } )
        indices_element = varargin{ 1 };
    else
        indices_element = num2cell( ones( size( xdc_arrays ) ) );
    end

	% ensure cell array for indices_element
	if ~iscell( indices_element )
        indices_element = { indices_element };
    end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( xdc_arrays, h_transfer, options_anti_aliasing, indices_element );

	%----------------------------------------------------------------------
	% 2.) compute anti-aliasing filter
	%----------------------------------------------------------------------
	% number of discrete frequencies
	N_samples_f = abs( [ h_transfer.axis ] );

	% specify cell array for h_samples_aa
	h_samples_aa = cell( size( xdc_arrays ) );

	% iterate transducer arrays
	for index_object = 1:numel( xdc_arrays )

        % check spatial anti-aliasing filter status
        if isa( options_anti_aliasing( index_object ), 'scattering.options_anti_aliasing_off' )

            %--------------------------------------------------------------
            % a) inactive spatial anti-aliasing filter
            %--------------------------------------------------------------
            h_samples_aa{ index_object } = h_transfer( index_object ).samples;

        elseif isa( options_anti_aliasing( index_object ), 'scattering.options_anti_aliasing_cosine' )

            %--------------------------------------------------------------
            % b) active spatial anti-aliasing filter w/ cosine roll-off
            %--------------------------------------------------------------
            % compute lateral components of mutual unit vectors
            N_dimensions_lateral = xdc_arrays( index_object ).N_dimensions;
            e_1_minus_2 = mutual_unit_vectors( math.grid( xdc_arrays( index_object ).positions_ctr ), h_transfer( index_object ).grid_FOV, indices_element{ index_object } );
            e_1_minus_2 = repmat( abs( e_1_minus_2( :, :, 1:(end - 1) ) ), [ N_samples_f( index_object ), 1 ] );

            % compute flag reflecting the local angular spatial frequencies
            axis_k_tilde = compute_wavenumbers( homogeneous_fluids.absorption_model, h_transfer( index_object ).axis );
            flag = real( axis_k_tilde.members ) .* e_1_minus_2 .* reshape( xdc_arrays( index_object ).cell_ref.edge_lengths, [ 1, 1, N_dimensions_lateral ] );

            indicator_no_aliasing = all( flag < pi, 3 );
% TODO: only for cosine
            indicator_taper = flag >= pi * ( 1 - options_anti_aliasing( index_object ).parameter );

            % compute anti-aliasing filter
            flag( ~indicator_taper ) = 1;
% TODO: small value of options_anti_aliasing( index_object ).parameter causes NaN
            flag( indicator_taper ) = cos( ( flag( indicator_taper ) - pi * ( 1 - options_anti_aliasing( index_object ).parameter ) ) / ( 2 * options_anti_aliasing( index_object ).parameter ) );
            flag = indicator_no_aliasing .* prod( flag, 3 );

            % apply anti-aliasing filter
            h_samples_aa{ index_object } = h_transfer( index_object ).samples .* flag;

        else

            %--------------------------------------------------------------
            % c) unknown options class
            %--------------------------------------------------------------
            errorStruct.message = sprintf( 'Class of options_anti_aliasing( %d ) is unknown!', index_object );
            errorStruct.identifier = 'anti_aliasing_filter:UnknownOptionsClass';
            error( errorStruct );

        end % if isa( options_anti_aliasing( index_object ), 'scattering.options_anti_aliasing_off' )

    end % for index_object = 1:numel( xdc_arrays )

    % create fields
    h_transfer_aa = discretizations.field( [ h_transfer.axis ], [ h_transfer.grid_FOV ], h_samples_aa );

end % function h_transfer_aa = anti_aliasing_filter( xdc_arrays, h_transfer, options_anti_aliasing, varargin )
