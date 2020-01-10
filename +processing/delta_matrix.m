%
% superclass for all delta matrices
%
% author: Martin F. Schiffner
% date: 2019-05-21
% modified: 2020-01-10
%
classdef delta_matrix < processing.signal_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = delta_matrix( indices_q, deltas, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices_q
            if ~iscell( indices_q )
                indices_q = { indices_q };
            end

            % class math.sequence_increasing_regular ensures class physical_values.physical_quantity for deltas

            % ensure nonempty weights
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                weights = varargin{ 1 };
            else
                weights = cell( size( indices_q ) );
                for index_object = 1:numel( indices_q )
                    weights{ index_object } = ones( size( indices_q{ index_object } ) );
                end
            end

            % ensure cell array for weights
            if ~iscell( weights )
                weights = { weights };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( indices_q, deltas, weights );

            %--------------------------------------------------------------
            % 2.) specify axes and samples
            %--------------------------------------------------------------
            % determine lower and upper integer bounds
            lbs_q = cellfun( @( x ) min( x( : ) ), indices_q );
            ubs_q = cellfun( @( x ) max( x( : ) ), indices_q );

            % specify regular axis
            axes = math.sequence_increasing_regular( lbs_q, ubs_q, deltas );
            N_samples = abs( axes );

            % determine numbers of signals
            N_signals = cellfun( @numel, indices_q );

            % specify cell arrays for samples
            samples = cell( size( indices_q ) );

            % iterate delta matrices
            for index_object = 1:numel( indices_q )

% TODO: check indices and weights

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( indices_q{ index_object }, weights{ index_object } );

                % indices of samples
                indices = ( 0:( N_signals( index_object ) - 1 ) )' * N_samples( index_object ) + indices_q{ index_object }( : ) + 1;

                % specify samples
                samples{ index_object } = zeros( N_samples( index_object ), N_signals( index_object ) ) * weights{ index_object }( 1 ) / deltas( index_object );
                samples{ index_object }( indices ) = weights{ index_object }( : ) / deltas( index_object );

            end % for index_object = 1:numel( indices_q )

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@processing.signal_matrix( axes, samples );

        end % function objects = delta_matrix( indices_q, deltas, varargin )

    end % methods

end % classdef delta_matrix < processing.signal_matrix
